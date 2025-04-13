# encoding=utf-8
"""
    Created on 21:29 2018/11/12
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class TCAGUI:
    def __init__(self, master):
        self.master = master
        master.title("迁移成分分析可视化系统 v1.0")
        master.geometry("1200x800")

        # 初始化变量
        self.src_path = tk.StringVar()
        self.tar_path = tk.StringVar()
        self.acc_var = tk.StringVar(value="准确率: --")
        self.params = {
            'kernel_type': tk.StringVar(value='linear'),
            'dim': tk.IntVar(value=30),
            'lamb': tk.DoubleVar(value=1.0),
            'gamma': tk.DoubleVar(value=1.0)
        }

        # 创建界面布局
        self.create_widgets()
        self.create_visualization()

    def create_widgets(self):
        """Create control panel"""
        control_frame = ttk.LabelFrame(self.master, text="Control Panel", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # File selection
        ttk.Button(control_frame, text="Select Source File",
                   command=lambda: self.select_file(self.src_path)).grid(row=0, column=0, pady=5)
        ttk.Entry(control_frame, textvariable=self.src_path, width=25).grid(row=0, column=1)

        ttk.Button(control_frame, text="Select Target File",
                   command=lambda: self.select_file(self.tar_path)).grid(row=1, column=0, pady=5)
        ttk.Entry(control_frame, textvariable=self.tar_path, width=25).grid(row=1, column=1)

        # Parameter settings
        params_frame = ttk.LabelFrame(control_frame, text="Model Parameters")
        params_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky='we')

        ttk.Label(params_frame, text="Kernel Type:").grid(row=0, column=0)
        kernel_combo = ttk.Combobox(params_frame, textvariable=self.params['kernel_type'],
                                    values=['linear', 'rbf', 'primal'])
        kernel_combo.grid(row=0, column=1)

        ttk.Label(params_frame, text="Dimension:").grid(row=1, column=0)
        ttk.Spinbox(params_frame, textvariable=self.params['dim'], from_=1, to=100).grid(row=1, column=1)

        ttk.Label(params_frame, text="Lambda:").grid(row=2, column=0)
        ttk.Entry(params_frame, textvariable=self.params['lamb']).grid(row=2, column=1)

        ttk.Label(params_frame, text="Gamma:").grid(row=3, column=0)
        ttk.Entry(params_frame, textvariable=self.params['gamma']).grid(row=3, column=1)

        # Action buttons
        ttk.Button(control_frame, text="Start Training", command=self.run_tca).grid(row=3, column=0, pady=10)
        ttk.Button(control_frame, text="Save Results", command=self.save_results).grid(row=3, column=1)

        # Result display
        ttk.Label(control_frame, textvariable=self.acc_var,
                  font=('Arial', 12, 'bold')).grid(row=4, column=0, columnspan=2)

    def create_visualization(self):
        """创建可视化区域"""
        vis_frame = ttk.Frame(self.master)
        vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建matplotlib图形
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        # 嵌入到tkinter窗口
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file(self, path_var):
        """文件选择对话框"""
        filename = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if filename:
            path_var.set(filename)

    def run_tca(self):
        """执行TCA训练和可视化"""
        try:
            # 加载数据
            src_domain = scipy.io.loadmat(self.src_path.get())
            tar_domain = scipy.io.loadmat(self.tar_path.get())

            # 获取数据（根据实际情况修改键名）
            Xs, Ys = src_domain['feas'], src_domain['labels'].ravel()
            Xt, Yt = tar_domain['feas'], tar_domain['labels'].ravel()

            # 初始化模型
            tca = TCA(
                kernel_type=self.params['kernel_type'].get(),
                dim=self.params['dim'].get(),
                lamb=self.params['lamb'].get(),
                gamma=self.params['gamma'].get()
            )

            # 训练并获取结果
            acc, ypred = tca.fit_predict(Xs, Ys, Xt, Yt)
            self.acc_var.set(f"Accuracy: {acc:.3f}")

            # 获取转换后的数据
            Xs_new, Xt_new = tca.fit(Xs, Xt)

            # 可视化
            self.visualize_data(Xs, Xt, Xs_new, Xt_new, Ys, Yt)

        except Exception as e:
            self.acc_var.set(f"错误: {str(e)}")

    def visualize_data(self, Xs, Xt, Xs_new, Xt_new, Ys, Yt):
        """数据可视化"""
        # 原始数据分布
        self.ax1.clear()
        X_combined = np.vstack([Xs, Xt])
        y_combined = np.hstack([Ys, Yt + max(Ys) + 1])  # 用不同颜色区分源域和目标域

        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_combined)

        # 绘制原始数据
        self.ax1.scatter(X_tsne[:len(Xs), 0], X_tsne[:len(Xs), 1],
                         c=Ys, cmap='tab10', label='Source')
        self.ax1.scatter(X_tsne[len(Xs):, 0], X_tsne[len(Xs):, 1],
                         c=Yt, cmap='tab10', marker='x', label='Target')
        self.ax1.set_title("Distribution of raw data (t-SNE)")
        self.ax1.legend()

        # 转换后数据分布
        self.ax2.clear()
        X_new_combined = np.vstack([Xs_new, Xt_new])
        X_new_tsne = tsne.fit_transform(X_new_combined)

        self.ax2.scatter(X_new_tsne[:len(Xs_new), 0], X_new_tsne[:len(Xs_new), 1],
                         c=Ys, cmap='tab10', label='Source')
        self.ax2.scatter(X_new_tsne[len(Xs_new):, 0], X_new_tsne[len(Xs_new):, 1],
                         c=Yt, cmap='tab10', marker='x', label='Target')
        self.ax2.set_title("TCA transformed data distribution (t-SNE)")
        self.ax2.legend()

        self.canvas.draw()

    def save_results(self):
        """保存可视化结果"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            self.fig.savefig(filename)


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)

        return acc, y_pred

    # TCA code is done here. You can ignore fit_new and fit_predict_new.

    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        # Computing projection matrix A from Xs an Xt
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot(
            [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]

        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1=Xt2, X2=X, gamma=self.gamma)

        # New target features
        Xt2_new = K @ A

        return Xt2_new

    def fit_predict_new(self, Xt, Xs, Ys, Xt2, Yt2):
        '''
        Transfrom Xt and Xs, get Xs_new
        Transform Xt2 with projection matrix created by Xs and Xt, get Xt2_new
        Make predictions on Xt2_new using classifier trained on Xs_new
        :param Xt: ns * n_feature, target feature
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt2: nt * n_feature, new target feature
        :param Yt2: nt * 1, new target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, _ = self.fit(Xs, Xt)
        Xt2_new = self.fit_new(Xs, Xt, Xt2)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt2_new)
        acc = sklearn.metrics.accuracy_score(Yt2, y_pred)

        return acc, y_pred


def train_valid():
    # If you want to perform train-valid-test, you can use this function
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in [1]:
        for j in [2]:
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(
                    src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']

                # Split target data
                Xt1, Xt2, Yt1, Yt2 = train_test_split(
                    Xt, Yt, train_size=50, stratify=Yt, random_state=42)

                # Create latent space and evaluate using Xs and Xt1
                tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt1, Yt1)

                # Project and evaluate Xt2 existing projection matrix and classifier
                acc2, ypre2 = tca.fit_predict_new(Xt1, Xs, Ys, Xt2, Yt2)

    print(f'Accuracy of mapped source and target1 data : {acc1:.3f}')  # 0.800
    print(f'Accuracy of mapped target2 data            : {acc2:.3f}')  # 0.706


if __name__ == '__main__':
    # Note: if the .mat file names are not the same, you can change them.
    # Note: to reproduce the results of my transfer learning book, use the dataset here: https://www.jianguoyun.com/p/DWJ_7qgQmN7PCBj29KsD (Password: cnfjmc)

    domains = ['./data/TCA/amazon_decaf.mat', './data/TCA/caltech_decaf.mat',
               './data/TCA/dslr_decaf.mat', './data/TCA/webcam_decaf.mat']

    for i in domains:
        for j in domains:
            if i != j:
                src, tar = i, j
                src_domain, tar_domain = scipy.io.loadmat(
                    src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['labels'], tar_domain['feas'], tar_domain['labels']
                tca = TCA(kernel_type='rbf', dim=30, lamb=1, gamma=1)
                acc, ypred = tca.fit_predict(Xs, Ys, Xt, Yt)
                print(i,'vs',j,f'Accuracy: {acc:.3f}')
    # root = tk.Tk()
    # app = TCAGUI(root)
    # root.mainloop()