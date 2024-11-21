import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE # type: ignore

# 方法1: 贝叶斯引导方法实现
def bayes_bootstrap(X, y, n_iterations):
    all_samples_X = []
    all_samples_y = []

    for _ in range(n_iterations):
        eta = np.random.dirichlet(np.ones(len(X)), 1).flatten()
        new_samples_X = []
        new_samples_y = []
        for i in range(len(X)):
            beta = (len(X) - 1) * eta[i]
            beta_int = int(beta)
            if i + beta_int + 1 < len(X):
                delta_X = X.iloc[i + beta_int + 1] - X.iloc[i]
                new_X = X.iloc[i] + (beta - beta_int) * delta_X
                new_samples_X.append(new_X)

                delta_y = y.iloc[i + beta_int + 1] - y.iloc[i]
                new_y = y.iloc[i] + (beta - beta_int) * delta_y
                new_samples_y.append(new_y)

        all_samples_X.extend(new_samples_X)
        all_samples_y.extend(new_samples_y)

    expanded_X = pd.DataFrame(all_samples_X, columns=X.columns)
    expanded_y = pd.Series(all_samples_y)
##保留了重复和空值
    # expanded_X.dropna(how='all', inplace=True)
    # expanded_y.dropna(inplace=True)

    expanded_X.reset_index(drop=True, inplace=True)
    expanded_y.reset_index(drop=True, inplace=True)

    # expanded_X.drop_duplicates(inplace=True)
    # expanded_y.drop_duplicates(inplace=True)

    return expanded_X, expanded_y

# 方法2: 扩充倍数的Bootstrap
def Bootstrap(X, y, expansion_factor):
    """
    扩充数据集，基于原始数据集进行 Bootstrap 扩充。
    :param X: 输入特征数据，DataFrame 格式
    :param y: 输入标签数据，Series 格式
    :param expansion_factor: 数据扩充倍数（即生成多少倍的数据）
    :return: 扩充后的特征和标签数据
    """
    n_samples = len(X)
    total_samples = int(n_samples * expansion_factor)  # 计算扩充后的样本总数
    bootstrap_samples_X = []
    bootstrap_samples_y = []

    for _ in range(total_samples):
        # 从原始数据集中随机抽样（带放回）
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X.iloc[bootstrap_indices]
        y_bootstrap = y.iloc[bootstrap_indices]
        
        # 将生成的样本添加到列表中
        bootstrap_samples_X.append(X_bootstrap)
        bootstrap_samples_y.append(y_bootstrap)

    # 将所有的样本合并成一个大的数据集
    expanded_X = pd.concat(bootstrap_samples_X, ignore_index=True)
    expanded_y = pd.concat(bootstrap_samples_y, ignore_index=True)

    # 返回扩充后的数据
    return expanded_X, expanded_y

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def neighbor_based_interpolation(features, targets, expansion_factor, n_neighbors=5):
    """
    基于最近邻的插值方法生成新样本，通过扩充倍数的方式生成新样本。
    :param features: 特征数据，DataFrame 格式
    :param targets: 目标标签数据，Series 格式
    :param expansion_factor: 扩充倍数，表示扩充的数据量是原数据的多少倍
    :param n_neighbors: 每个样本的邻居数
    :return: 新生成的特征和标签数据
    """
    # 训练最近邻模型
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(features)

    # 找到每个样本的最近邻
    distances, indices = nn.kneighbors(features)

    # 计算需要生成的样本数量
    num_samples = int(len(features) * (expansion_factor - 1))  # 扩充倍数 - 1 是新增的样本数

    new_features_list = []
    new_targets_list = []

    # 使用加权均值生成新样本
    epsilon = 1e-8
    for _ in range(num_samples):
        for i in range(len(features)):
            # 计算加权均值
            weights = 1 / (distances[i] + epsilon)
            weighted_feature_sum = np.average(features.iloc[indices[i]], axis=0, weights=weights)
            weighted_target_sum = np.average(targets.iloc[indices[i]], axis=0, weights=weights)

            new_features_list.append(weighted_feature_sum)
            new_targets_list.append(weighted_target_sum)

            if len(new_features_list) >= num_samples:
                return pd.DataFrame(new_features_list, columns=features.columns), pd.Series(new_targets_list)



# 方法2: SMOTE 随机过采样
from imblearn.over_sampling import SMOTEN

def generate_regressdata_SMOTE(X, y, get_multiple):
    smote = SMOTEN(sampling_strategy=get_multiple, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

# 方法3: GAN生成对抗网络
# 使用GAN生成对抗网络进行数据增强
def generate_regressdata_GAN(X_train, y_train, get_multiple=5):
    def build_generator(input_dim, output_dim):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_dim=input_dim, activation='relu'))
        model.add(tf.keras.layers.Dense(output_dim, activation='linear'))  # 对应回归的连续值输出
        return model

    def build_discriminator(input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_dim=input_shape[1], activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    def train_gan(X_train, y_train, batch_size, epochs, get_multiple):
        input_dim = 100
        output_dim = X_train.shape[1]
        discriminator = build_discriminator(X_train.shape)
        generator = build_generator(input_dim, output_dim)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

        discriminator.trainable = False
        gan_input = tf.keras.layers.Input(shape=(input_dim,))
        generated_data = generator(gan_input)
        gan_output = discriminator(generated_data)
        gan = tf.keras.models.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)

        for epoch in range(epochs):
            for _ in range(get_multiple):
                noise = np.random.normal(0, 1, (batch_size, input_dim))
                real_data = X_train.iloc[np.random.randint(0, X_train.shape[0], size=batch_size)]
                real_labels = np.ones((batch_size, 1))

                generated_data = generator.predict(noise)
                fake_labels = np.zeros((batch_size, 1))

                d_loss_real = discriminator.train_on_batch(real_data, real_labels)
                d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)

                noise = np.random.normal(0, 1, (batch_size, input_dim))
                g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        return generator

    generator = train_gan(X_train, y_train, batch_size=128, epochs=100, get_multiple=get_multiple)

    noise_samples = np.random.normal(0, 1, (get_multiple * len(X_train), 100))
    synthetic_data = generator.predict(noise_samples)

    synthetic_labels = np.mean(y_train) + np.std(y_train) * np.random.normal(size=(len(synthetic_data),))

    return synthetic_data, synthetic_labels



# 方法4: GMM高斯混合模型
def generate_regressdata_GMM(origin_data, origin_data_label, get_multiple):
    no_of_synthetic = get_multiple * len(origin_data_label)
    gmm = GaussianMixture(n_components=1)
    gmm.fit(origin_data)

    synthetic_data = gmm.sample(no_of_synthetic)[0]
    synthetic_labels = np.mean(origin_data_label) + np.std(origin_data_label) * np.random.randn(no_of_synthetic)

    return synthetic_data, synthetic_labels


# 方法5: LSTM长短时记忆网络 

def generate_regressdata_LSTM(origin_data, origin_data_label, get_multiple):
    origin_data = origin_data.to_numpy().reshape(-1, 1)  # 转换为 numpy 数组并调整形状
    scaler = StandardScaler()
    data = scaler.fit_transform(origin_data)

    model = Sequential([
        LSTM(128, input_shape=(None, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    synthetic_data_list = []
    synthetic_labels = []

    # 生成合成数据
    for _ in range(get_multiple):
        X_train = data[:-1].reshape(1, -1, 1)  # 重塑训练数据
        Y_train = data[1:].reshape(1, -1, 1)

        model.fit(X_train, Y_train, epochs=40, batch_size=64, verbose=0)

        synthetic_data = model.predict(X_train).reshape(-1)  # 预测并重塑
        synthetic_data = scaler.inverse_transform(synthetic_data.reshape(-1, 1)).reshape(-1)  # 反归一化
        synthetic_data_list.append(synthetic_data)

        # 确保每次扩展的标签数量与合成数据数量一致
        synthetic_labels.extend([origin_data_label] * len(synthetic_data))  # 扩展标签

    # 将合成数据列表转为 numpy 数组
    synthetic_data_combined = np.vstack(synthetic_data_list)
    synthetic_labels = np.array(synthetic_labels)

    # 确保合成数据和标签的样本数一致
    if synthetic_data_combined.shape[0] != synthetic_labels.shape[0]:
        raise ValueError("合成数据和标签的样本数不匹配")

    # 这里需要确保返回的数据是合适的形状
    return synthetic_data_combined.reshape(-1, 1), synthetic_labels.reshape(-1, 1)


