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

# 方法2: SMOTE 随机过采样
def generate_regressdata_SMOTE(X, y, get_multiple):
    unique_classes = np.unique(y)
    x_aug = np.copy(X)
    y_aug = np.copy(y)
    
    for label in unique_classes:
        num_minority = np.sum(y == label)
        smote = SMOTE(sampling_strategy={label: num_minority * get_multiple}, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X[y == label], y[y == label])
        x_aug = np.vstack((x_aug, X_resampled))
        y_aug = np.hstack((y_aug, y_resampled))
    
    return x_aug, y_aug

# 方法3: GAN生成对抗网络
# 使用GAN生成对抗网络进行数据增强
def generate_classdata_GAN(X_train, y_train, get_multiple=5):
    # 构建生成器
    def build_generator(input_dim, output_dim):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_dim=input_dim, activation='relu'))
        model.add(tf.keras.layers.Dense(output_dim, activation='tanh'))
        return model

    # 构建判别器
    def build_discriminator(input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_dim=input_shape[1], activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    # 训练GAN模型
    def train_gan(X_train, batch_size, epochs, get_multiple):
        input_dim = 100  # 生成器的输入维度
        output_dim = X_train.shape[1]  # 生成器输出维度与数据集特征维度一致
        discriminator = build_discriminator(X_train.shape)
        generator = build_generator(input_dim, output_dim)

        # 编译判别器
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # 创建GAN模型
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

    # GAN模型训练
    batch_size = 128  # 每次训练的样本数
    epochs = 100  # 训练轮数
    generator = train_gan(X_train, batch_size, epochs, get_multiple)

    # 生成新样本
    noise_samples = np.random.normal(0, 1, (get_multiple * len(X_train), 100))
    synthetic_data = generator.predict(noise_samples)

    # 返回生成的数据和标签
    return synthetic_data, np.full((synthetic_data.shape[0],), y_train[0])




# 方法4: GMM高斯混合模型
def generate_classdata_GMM(origin_data, origin_data_label, get_multiple):
    no_of_synthetic = get_multiple * len(origin_data_label)
    
    gmm = GaussianMixture(n_components=len(np.unique(origin_data_label)))
    gmm.fit(origin_data)
    
    synthetic_data = gmm.sample(no_of_synthetic)[0]
    
    kmeans = KMeans(n_clusters=len(np.unique(origin_data_label)))
    synthetic_label = kmeans.fit_predict(synthetic_data)
    
    return synthetic_data, synthetic_label

# 方法5: LSTM长短时记忆网络
def generate_classdata_LSTM(origin_data, origin_data_label, get_multiple):
    # 将 origin_data 转换为 NumPy 数组，并确保它有合适的形状
    origin_data = origin_data.to_numpy().reshape(-1, 1)  # 修改此处，将 DataFrame 转换为 NumPy 数组
    scaler = StandardScaler()
    data = scaler.fit_transform(origin_data)

    num_features = 1
    num_hidden_units = 128
    num_responses = 1
    num_epochs = 40
    batch_size = 64

    model = Sequential([
        LSTM(num_hidden_units, input_shape=(None, num_features)),
        Dense(num_responses)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    synthetic_data_list = []
    for _ in range(get_multiple):
        X_train = data[:-1].reshape(1, -1, 1)  # 确保数据有适当的形状 (samples, timesteps, features)
        Y_train = data[1:].reshape(1, -1, 1)

        model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)

        synthetic_data = model.predict(X_train).reshape(-1)
        synthetic_data = scaler.inverse_transform(synthetic_data.reshape(-1, 1)).reshape(-1)

        synthetic_data_list.append(synthetic_data)

    synthetic_data_combined = np.vstack(synthetic_data_list)
    synthetic_label = np.repeat(origin_data_label, get_multiple)

    return synthetic_data_combined, synthetic_label
