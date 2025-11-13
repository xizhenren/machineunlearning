from scipy.optimize import minimize
from Unlearner.LRUnlearner import LogisticRegressionUnlearner
from scipy.special import expit
import numpy as np
import scipy.sparse as sp
import time
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
from scipy.linalg import inv
import json

#new-----
from Unlearner.Wasserstein import wdp_sigma_from_sensitivity

# DPLRUnlearner类实现了一种 差分隐私逻辑回归遗忘算法（Differentially Private Logistic Regression Unlearning）
class DPLRUnlearner(LogisticRegressionUnlearner): #继承了 LogisticRegressionUnlearner
    def __init__(self, train_data, test_data, voc, epsilon, delta, sigma, lambda_, b=None, category_to_idx_dict=None, mu: float = 1.0, wdp_target_epsilon: float | None = None, delta_relax=1e-5,wdp_mode: bool = True):
        self.set_train_test_data(train_data, test_data) #训练与测试数据集（已分好X, y）
        self.lambda_ = lambda_ #L2 正则化系数
        self.feature2dim = voc #特征字典，映射特征 → 维度索引
        self.dim2feature = {v:k for k,v in self.feature2dim.items()}
        self.epsilon = epsilon #差分隐私预算参数
        self.delta = delta
        self.sigma = sigma #高斯噪声标准差，用于隐私保护
        self.lambda_ = lambda_
        self.voc = voc
        self.category_to_idx_dict = category_to_idx_dict  #用于分类特征随机替换的索引字典
        self.theta = np.random.standard_normal(self.x_train.shape[1])  # sample weights normal distributed 初始化模型参数
        self.model_param_str = 'lambda={}_epsilon={}_delta={}_sigma={}'.format(   #生成模型标识字符串
            self.lambda_, self.epsilon, self.delta, self.sigma)
        
        #new ------------
        self.mu = mu
        self.wdp_target_epsilon = wdp_target_epsilon
        self.wdp_mode = wdp_mode
        self.delta_relax = delta_relax  # 松弛项

        # === WDP: 如需要，用 Δ2 和目标 epsilon 标定 sigma ===
        if self.wdp_mode and self.wdp_target_epsilon is not None and (sigma is None or sigma == 0):
            # 一个轻量的 Δ2 估计：用二阶近似 H^{-1} * (∑ ∇ℓ(tilde z) - ∑ ∇ℓ(z)) 的 L2 范数
            # 这里我们用“单点最坏情况”的保守上界：任选一个样本 i 计算梯度差的范数上界。
            # 工程上你也可以传入具体的 indices_to_delete，在调用处更精确地估。
            i0 = 0
            g_i0 = self.get_gradient_l(self.theta, self.x_train[i0:i0+1], self.y_train[i0:i0+1])
            # 删除/翻转一个样本的梯度差可用 2*|g_i0| 作保守上界
            #（如你已有更精确的一阶/二阶闭式更新，可替换成那部分的范数）
            delta_grad_norm = np.linalg.norm(g_i0.reshape(-1), ord=2)
            # 近似 Δ2 ≈ || H^{-1} || * || Δgrad ||，用逆 Hessian 的谱范数上界
            H_inv = self.get_inverse_hessian(self.x_train, self.theta)
            # 谱范数上界（保守）：用 ||H^{-1}||_2 ≤ sqrt(||H^{-1}||_F^2)
            H_inv_op_norm = np.linalg.norm(H_inv, ord=2)
            delta2 = float(H_inv_op_norm * (2.0 * delta_grad_norm))

            self.sigma = wdp_sigma_from_sensitivity(delta2, self.wdp_target_epsilon, self.mu, self.delta_relax)
        else:
            self.sigma = sigma
        
        #-----------------------------


        if b is None: #外部提供的噪声向量；若为空则根据 σ 生成
            self.b = np.random.normal(0, self.sigma, size=self.x_train.shape[1]) if self.sigma != 0 else np.zeros(self.x_train.shape[1])
        else:
            self.b = b
        self.gradient_calls = 0

        

    # computes l(x,y;theta). if x and y contain multiple samples l is summed up over them
    # we use l(x,y) = log(1+exp(-y*theta^T*x))
    # 单样本/批样本的逻辑回归损失 计算样本的逻辑损失，不含正则化与隐私项
    def get_loss_l(self, theta, x, y):
        dot_prod = x.dot(theta) * y
        data_loss = -np.log(expit(dot_prod))
        total_loss = np.sum(data_loss, axis=0)
        return total_loss

    # computes L(x,y;theta)
    # 计算样本的完整逻辑损失（含正则化与隐私项）
    # def get_loss_L(self, theta, x, y):
    #     summed_loss = self.get_loss_l(theta, x, y)
    #     total_loss = summed_loss + 0.5*self.lambda_*np.dot(theta, theta.T) + np.dot(self.b, theta)
    #     return total_loss
    

    def get_loss_L(self, theta, x, y, theta_retrained=None):
        summed_loss = self.get_loss_l(theta, x, y)

        if theta_retrained is not None:
            # 计算Wasserstein距离（L2范数）
            wasserstein_distance = np.linalg.norm(theta.reshape(-1) - theta_retrained.reshape(-1), ord=2)
                
            # 动态调整lambda_，可以使用Wasserstein距离作为调整因子
            lambda_dynamic =1 - (wasserstein_distance / 15)  # 例如，使用指数函数调节lambda_
        else:
            lambda_dynamic = 0.5  # 如果没有提供theta_retrained，保持原始lambda_


        total_loss = summed_loss + lambda_dynamic*self.lambda_*np.dot(theta, theta.T) + np.dot(self.b, theta)
        return total_loss

    # return total loss L on train set.
    # def get_train_set_loss(self, theta):
    #     return self.get_loss_L(theta, self.x_train, self.y_train)

    def get_train_set_loss(self, theta, theta_retrained=None):
        return self.get_loss_L(theta, self.x_train, self.y_train, theta_retrained)

    # get gradient w.r.t. parameters (-y*x*sigma(-y*Theta^Tx)) for y in {-1,1}
    # 梯度计算模块,计算单样本梯度
    def get_gradient_l(self, theta, x, y):
        assert x.shape[0] == y.shape[0], f'{x.shape[0]} != {y.shape[0]}'
        dot_prod = x.dot(theta) * y
        factor = -expit(-dot_prod) * y
        # we need to multiply every row of x by the corresponding value in factor vector
        if type(x) is sp.csr_matrix:
            factor_m = sp.diags(factor)
            res = factor_m.dot(x)
        else:
            res = np.expand_dims(factor, axis=1) * x
        grad = res.sum(axis=0)
        if type(grad) is np.matrix:
            grad = grad.A
        return grad
    
    # 计算带正则与噪声项的总梯度
    def get_gradient_L(self, theta, x, y):
        summed_grad = self.get_gradient_l(theta, x, y)
        total_grad = summed_grad + self.lambda_ * theta + self.b
        total_grad = total_grad.squeeze()
        self.gradient_calls += 1
        return total_grad

    # this is the gradient of L on the train set. This should be close to zero after fitting.
    # 在训练集上计算完整梯度（用于优化终止条件）
    def get_train_set_gradient(self, theta):
        return self.get_gradient_L(theta, self.x_train, self.y_train)

    # computes inverse hessian for data x. As we only need the inverse hessian on the entire dataset we return the
    # Hessian on the full L loss.
    # 计算 Hessian 矩阵并求逆
    def get_inverse_hessian(self, x, theta=None):
        if theta is None:
            theta = self.theta
        dot = x.dot(theta)
        probs = expit(dot)
        weighting = probs * (1-probs) # sigma(-t) = (1-sigma(t))
        if type(x) is sp.csr_matrix:
            weighting_m = sp.diags(weighting)
            p1 = x.transpose().dot(weighting_m)
        else:
            p1 = x.transpose() * np.expand_dims(weighting, axis=0)
        res = p1.dot(x)
        res += self.lambda_ * np.eye(self.dim)  # hessian of regularization
        cov_inv = inv(res)
        return cov_inv
    
    # 一阶近似更新
    def get_first_order_update(self, G, unlearning_rate, theta=None):
        if theta is None:
            theta = self.theta
        return theta - unlearning_rate * G
    
    # 二阶近似更新（用 Hessian 逆矩阵）
    def get_second_order_update(self, x, y, G, theta=None):
        if theta is None:
            theta = self.theta
        H_inv = self.get_inverse_hessian(x, theta)
        return theta - np.dot(H_inv, G)
    
    # 分步更新策略：用于逐步删除样本或特征时，根据“删除”或“替换”模式执行多次一阶或二阶更新。
    def get_order_update_stepwise(self, indices, stepsize, remove, n_replacements, order, unlearning_rate=None):
        if order == 1:
            assert unlearning_rate is not None
        l_indices = len(indices)
        theta_tmp = self.theta.copy()
        x_tmp = self.x_train.copy()
        if remove:
            for idx in range(0, l_indices, stepsize):
                indices_to_remove = indices[idx:idx+stepsize]
                x_delta, changed_rows = self.copy_and_replace(x_tmp, indices_to_remove, remove)
                z_tmp = (x_tmp[changed_rows], self.y_train[changed_rows])
                z_delta_tmp = (x_delta[changed_rows], self.y_train[changed_rows])
                G_tmp = self.get_G(z_tmp, z_delta_tmp, theta_tmp)
                if order == 1:
                    theta_tmp = self.get_first_order_update(G_tmp, unlearning_rate, theta=theta_tmp)
                else:
                    theta_tmp = self.get_second_order_update(x_delta, self.y_train, G_tmp, theta=theta_tmp)
                x_tmp = x_delta.copy()
        else:
            replacements_per_round = n_replacements // stepsize
            for i in range(stepsize):
                x_delta, changed_rows = self.copy_and_replace(x_tmp, indices, remove, replacements_per_round)
                z_tmp = (x_tmp[changed_rows], self.y_train[changed_rows])
                z_delta_tmp = (x_delta[changed_rows], self.y_train[changed_rows])
                G_tmp = self.get_G(z_tmp, z_delta_tmp, theta_tmp)
                if order == 1:
                    theta_tmp = self.get_first_order_update(G_tmp, unlearning_rate, theta=theta_tmp)
                else:
                    theta_tmp = self.get_second_order_update(x_delta, self.y_train, G_tmp, theta=theta_tmp)
                x_tmp = x_delta.copy()

        return theta_tmp, x_tmp

    def get_fine_tuning_update(self, x, y, learning_rate, batch_size=32):
        new_theta = self.theta.copy()
        for i in range(0, x.shape[0], batch_size):
            grad = self.get_gradient_L(new_theta, x[i:i+batch_size], y[i:i+batch_size])
            new_theta -= 1./batch_size * learning_rate * grad
        return new_theta

    # given indices_to_delete (i.e. column indices) computes row indices where the column indices are non-zero
    def get_relevant_indices(self, indices_to_delete):
        # get the rows (samples) where the features appear
        relevant_indices = self.x_train[:, indices_to_delete].nonzero()[0]
        # to avoid having samples more than once
        relevant_indices = np.unique(relevant_indices)
        return relevant_indices
    
    # 计算梯度变化：即在删除/替换后数据的梯度差，用于近似遗忘更新。
    def get_G(self, z, z_delta, theta=None):
        """
        Computes G as defined in the paper using z=(x,y) and z_delta=(x_delta, y_delta)
        :param z: Tuple of original (unchanged) data (np.array /csr_matrix , np.array)
        :param z_delta: Tuple of changed data (np.array /csr_matrix , np.array)
        :return: G=\sum \nabla l(z_delta)-\nabla l(z)
        """
        if theta is None:
            theta = self.theta
        grad_z_delta = self.get_gradient_l(theta, z_delta[0], z_delta[1])
        grad_z = self.get_gradient_l(theta, z[0], z[1])
        diff = grad_z_delta - grad_z
        if type(z[0]) is sp.csr_matrix:
            diff = diff.squeeze()
        return diff
    
    # Sigmoid 判定
    def predict(self, x, theta):
        logits = expit(x.dot(theta))
        y_pred = np.array([1 if l >= 0.5 else -1 for l in logits])
        return y_pred
    
    # 输出分类报告、AUC、Loss、梯度范数等指标
    def get_performance(self, x, y, theta):
        assert x.shape[0] == y.shape[0], '{} != {}'.format(x.shape[0], y.shape[0])
        logits = expit(x.dot(theta))
        y_pred = np.array([1 if l >= 0.5 else -1 for l in logits])
        accuracy = len(np.where(y_pred == y)[0])/x.shape[0]
        fpr, tpr, _ = roc_curve(y, logits)
        prec, rec, _ = precision_recall_curve(y, logits)
        auc_roc = auc(fpr, tpr)
        auc_pr = auc(rec, prec)
        report = classification_report(y, y_pred, digits=4, output_dict=True)
        n_data = x.shape[0]
        loss = self.get_loss_L(theta, x, y)
        grad = self.get_gradient_L(theta, x, y)
        report['test_loss'] = loss
        report['gradient_norm'] = np.sum(grad**2)
        report['train_loss'] = self.get_train_set_loss(theta)
        report['gradient_norm_train'] = np.sum(self.get_train_set_gradient(theta)**2)
        report['accuracy'] = accuracy
        report['test_roc_auc'] = auc_roc
        report['test_pr_auc'] = auc_pr
        return report
    
    # 用 L-BFGS-B 优化器训练逻辑回归。输出：训练耗时、精度、梯度残差等
    def fit_model(self):
        start_time = time.time()
        #res = minimize(self.get_train_set_loss, self.theta, method='L-BFGS-B', jac=self.get_train_set_gradient,
        #               options={'disp':True})
        res = minimize(self.get_train_set_loss, self.theta, method='L-BFGS-B', jac=self.get_train_set_gradient,
                       options={'maxiter': 1000})
        end_time = time.time()
        total_time = end_time-start_time
        self.theta = res.x
        #print(f'Fitting took {total_time} seconds.')
        performance = self.get_performance(self.x_test, self.y_test, self.theta)
        acc = performance['accuracy']
        gr = performance['gradient_norm_train']
        #print(f'Achieved accuracy: {acc}')
        #print(f'Gradient residual train: {gr}')
        #print(json.dumps(performance, indent=4))

    # 返回参数绝对值最大的 n 个特征（解释性分析）
    def get_n_largest_features(self, n):
        theta_abs = np.abs(self.theta)
        largest_features_ind = np.argsort(-theta_abs)[:n]
        largest_features = [self.dim2feature[d] for d in largest_features_ind]
        return largest_features_ind, largest_features

    #@staticmethod
    # 核心“遗忘操作”：
    # 若 remove=True，则将指定特征列置零；
    # 若为替换模式，则：对分类特征随机重置类别；对连续特征设为 0 或随机值
    def copy_and_replace(self, x, indices, remove=False, n_replacements=0):
        """
        Helper function that sets 'indices' in 'arr' to 'value'
        :param x - numpy array or csr_matrix of shape (n_samples, n_features)
        :param indices - the columns where the replacement should take place
        :param remove - if true the entire columns will be deleted (set to zero). Otherwise values will be set to random value
        :param n_replacements - if remove is False one can specify how many samples are adjusted.
        :return copy of arr with changes, changed row indices
        """
        x_cpy = x.copy()
        if sp.issparse(x):
            x_cpy = x_cpy.tolil()
        if remove:
            relevant_indices = x_cpy[:, indices].nonzero()[0]
            # to avoid having samples more than once
            relevant_indices = np.unique(relevant_indices)
            x_cpy[:, indices] = 0
        else:
            relevant_indices = np.random.choice(x_cpy.shape[0], n_replacements, replace=False)
            unique_indices = set(np.unique(x_cpy[:, indices]).tolist())
            if unique_indices == {0, 1}:
                voc_rev = {v:k for k,v in self.voc.items()}
                # if we have only binary features we need the category dict
                #x_cpy[np.ix_(relevant_indices, indices)] = - 1 * x_cpy[np.ix_(relevant_indices, indices)] + 1
                for ri in relevant_indices:
                    for category, category_columns in self.category_to_idx_dict.items():
                        #print(f'Processing {category}')
                        category_data = x_cpy[ri, category_columns]
                        # check if the category is set
                        ones = np.where(category_data == 1)[0]
                        if len(ones) > 0:
                            assert len(ones) == 1
                            ind_to_delete = category_columns[ones[0]]
                            #print(f'Found category {voc_rev[ind_to_delete]}')
                            x_cpy[ri, ind_to_delete] = 0
                            if len(category_columns) > 1:
                                list_to_choose_from = [i for i in category_columns if i != ind_to_delete]
                            else:
                                list_to_choose_from = []
                        else:
                            list_to_choose_from = category_columns
                        if len(list_to_choose_from) > 0:
                            col_to_set = np.random.choice(list_to_choose_from)
                            #print(f'Set {voc_rev[col_to_set]} to one')
                            x_cpy[ri, col_to_set] = 1

            else:
                # else we choose random values
                for idx in indices:
                    #random_values = np.random.choice(x_cpy[:, idx], n_replacements, replace=False)
                    #x_cpy[relevant_indices, idx] = random_values
                    mean = np.mean(x_cpy[:, idx])
                    std = np.std(x_cpy[:,idx])
                    #x_cpy[relevant_indices, idx] = np.random.normal(0,6*std)
                    x_cpy[relevant_indices, idx] = np.zeros(relevant_indices.shape)
        if sp.issparse(x):
            x_cpy = x_cpy.tocsr()
        return x_cpy, relevant_indices