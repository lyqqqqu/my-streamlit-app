import numpy as np
import GPy
from sklearn.metrics import r2_score, mean_squared_error
from deap import base, creator, tools, algorithms


def ga_optimize(evaluate_fitness, lb, ub, max_gen=30, pop_size=20, cxpb=0.5, mutpb=0.2):
    """
    通用GA优化函数。

    参数:
    - evaluate_fitness: callable, 接受一个表示权重的数组或列表，返回适应度值（越小越好）
    - lb: list, 参数下界
    - ub: list, 参数上界
    - max_gen: int, 最大迭代次数（代数）
    - pop_size: int, 种群大小
    - cxpb: float, 交叉概率
    - mutpb: float, 变异概率

    返回:
    - best_solution: ndarray, 最优解
    - iteration_bests: list, 每一代的最佳适应度记录
    """

    # 定义适应度函数类型（最小化问题）
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # 初始化个体
    def init_ind(icls):
        return icls([np.random.uniform(lb[i], ub[i]) for i in range(len(lb))])

    toolbox.register("individual", init_ind, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (evaluate_fitness(ind),))
    toolbox.register("mate", tools.cxTwoPoint)

    # 使用高斯变异或其他连续变异算子。这里示意用高斯变异:
    # 注意sigma可以根据搜索空间大小调整
    sigma = [(ub[i] - lb[i]) * 0.1 for i in range(len(lb))]  # 假设sigma为搜索空间的10%
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=0.1)

    toolbox.register("select", tools.selTournament, tournsize=3)

    # 初始化种群
    pop = toolbox.population(n=pop_size)
    iteration_bests = []
    best_fit_so_far = float("inf")

    for gen in range(max_gen):
        # 评估本代适应度
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # 获取本代最优
        fits = [ind.fitness.values[0] for ind in pop]
        gen_best = min(fits)
        if gen_best < best_fit_so_far:
            best_fit_so_far = gen_best
        iteration_bests.append(best_fit_so_far)

        if gen < max_gen - 1:
            # 选择
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            # 变异
            for mut_ind in offspring:
                if np.random.random() < mutpb:
                    toolbox.mutate(mut_ind)
                    del mut_ind.fitness.values
                    # 确保变异后仍在上下界内
                    for i in range(len(mut_ind)):
                        mut_ind[i] = max(lb[i], min(ub[i], mut_ind[i]))

            pop[:] = offspring

    # 找到最终最优解
    final_fits = [ind.fitness.values[0] for ind in pop]
    best_ind = pop[np.argmin(final_fits)]
    best_solution = np.array(best_ind)

    return best_solution, iteration_bests