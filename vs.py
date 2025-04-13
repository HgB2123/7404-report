# encoding=utf-8
"""
迁移学习对比分析系统（TCA + KMM）
Created on 2023/10/15
@author: AI助手
"""
import numpy as np
import scipy.io
import scipy.linalg
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from cvxopt import matrix, solvers


class TCAGUI:
    def __init__(self, master):
        self.master = master
        master.title("Transfer Learning Analysis System v2.0")
        master.geometry("1400x800")

        # Initialize variables
        self.src_path = tk.StringVar()
        self.tar_path = tk.StringVar()
        self.acc_var = tk.StringVar(value="TCA Accuracy: -- | KMM Accuracy: --")
        self.params = {
            # TCA parameters
            'tca_kernel': tk.StringVar(value='linear'),
            'tca_dim': tk.IntVar(value=30),
            'tca_lamb': tk.DoubleVar(value=1.0),
            'tca_gamma': tk.DoubleVar(value=1.0),
            # KMM parameters
            'kmm_kernel': tk.StringVar(value='rbf'),
            'kmm_B': tk.DoubleVar(value=10.0),
            'kmm_eps': tk.DoubleVar(value=0.2),
            'kmm_gamma': tk.DoubleVar(value=1.0)
        }

        # Create UI components
        self.create_widgets()
        self.create_visualization()
        self.beta = None  # Store KMM weights

    def create_widgets(self):
        """Create control panel"""
        control_frame = ttk.LabelFrame(self.master, text="Control Panel", width=320)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # File selection
        ttk.Button(control_frame, text="Select Source Domain",
                   command=lambda: self.select_file(self.src_path)).grid(row=0, column=0, pady=5)
        ttk.Entry(control_frame, textvariable=self.src_path, width=28).grid(row=0, column=1)

        ttk.Button(control_frame, text="Select Target Domain",
                   command=lambda: self.select_file(self.tar_path)).grid(row=1, column=0, pady=5)
        ttk.Entry(control_frame, textvariable=self.tar_path, width=28).grid(row=1, column=1)

        # TCA parameters
        tca_frame = ttk.LabelFrame(control_frame, text="TCA Parameters")
        tca_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky='we')

        ttk.Label(tca_frame, text="Kernel Type:").grid(row=0, column=0)
        ttk.Combobox(tca_frame, textvariable=self.params['tca_kernel'],
                     values=['linear', 'rbf', 'primal']).grid(row=0, column=1)

        ttk.Label(tca_frame, text="Dimension:").grid(row=1, column=0)
        ttk.Spinbox(tca_frame, textvariable=self.params['tca_dim'],
                    from_=1, to=100).grid(row=1, column=1)

        ttk.Label(tca_frame, text="Lambda:").grid(row=2, column=0)
        ttk.Entry(tca_frame, textvariable=self.params['tca_lamb']).grid(row=2, column=1)

        ttk.Label(tca_frame, text="Gamma:").grid(row=3, column=0)
        ttk.Entry(tca_frame, textvariable=self.params['tca_gamma']).grid(row=3, column=1)

        # KMM parameters
        kmm_frame = ttk.LabelFrame(control_frame, text="KMM Parameters")
        kmm_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky='we')

        ttk.Label(kmm_frame, text="Kernel Type:").grid(row=0, column=0)
        ttk.Combobox(kmm_frame, textvariable=self.params['kmm_kernel'],
                     values=['linear', 'rbf']).grid(row=0, column=1)

        ttk.Label(kmm_frame, text="B Value:").grid(row=1, column=0)
        ttk.Entry(kmm_frame, textvariable=self.params['kmm_B']).grid(row=1, column=1)

        ttk.Label(kmm_frame, text="Epsilon:").grid(row=2, column=0)
        ttk.Entry(kmm_frame, textvariable=self.params['kmm_eps']).grid(row=2, column=1)

        ttk.Label(kmm_frame, text="Gamma:").grid(row=3, column=0)
        ttk.Entry(kmm_frame, textvariable=self.params['kmm_gamma']).grid(row=3, column=1)

        # Action buttons
        ttk.Button(control_frame, text="Start Analysis", command=self.run_analysis,
                   style='Accent.TButton').grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(control_frame, text="Save Results", command=self.save_results).grid(row=5, column=0, columnspan=2)

        # Result display
        ttk.Label(control_frame, textvariable=self.acc_var,
                  font=('Arial', 11, 'bold')).grid(row=6, column=0, columnspan=2)

    def create_visualization(self):
        """Create visualization area"""
        vis_frame = ttk.Frame(self.master)
        vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create three-column visualization layout
        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(131)  # Original data
        self.ax2 = self.fig.add_subplot(132)  # TCA result
        self.ax3 = self.fig.add_subplot(133)  # KMM result

        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file(self, path_var):
        """File selection dialog"""
        filename = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if filename:
            path_var.set(filename)

    def run_analysis(self):
        """Execute analysis pipeline"""
        try:
            # Load data
            src_domain = scipy.io.loadmat(self.src_path.get())
            tar_domain = scipy.io.loadmat(self.tar_path.get())

            # Get data (adjust key names according to actual data)
            Xs, Ys = src_domain['feas'], src_domain['labels'].ravel()
            Xt, Yt = tar_domain['feas'], tar_domain['labels'].ravel()

            # Execute both methods
            tca_acc, kmm_acc, Xs_tca, Xt_tca, Xs_kmm = self.run_methods(Xs, Ys, Xt, Yt)

            # Update UI
            self.acc_var.set(f"TCA Accuracy: {tca_acc:.3f} | KMM Accuracy: {kmm_acc:.3f}")
            self.visualize_all(Xs, Xt, Xs_tca, Xt_tca, Xs_kmm, Ys, Yt)

        except Exception as e:
            self.acc_var.set(f"Error: {str(e)}")
            raise e

    def run_methods(self, Xs, Ys, Xt, Yt):
        """Execute TCA and KMM methods"""
        # TCA processing
        tca = TCA(
            kernel_type=self.params['tca_kernel'].get(),
            dim=self.params['tca_dim'].get(),
            lamb=self.params['tca_lamb'].get(),
            gamma=self.params['tca_gamma'].get()
        )
        tca_acc, _ = tca.fit_predict(Xs, Ys, Xt, Yt)
        Xs_tca, Xt_tca = tca.fit(Xs, Xt)

        # KMM processing (with dimension adjustment)
        kmm = KMM(
            kernel_type=self.params['kmm_kernel'].get(),
            B=self.params['kmm_B'].get(),
            eps=self.params['kmm_eps'].get(),
            gamma=self.params['kmm_gamma'].get()
        )
        self.beta = kmm.fit(Xs, Xt)

        # Reshape beta to column vector (n_samples, 1)
        beta_reshaped = self.beta.reshape(-1, 1)

        # Perform per-sample weighting
        Xs_kmm = beta_reshaped * Xs  # Correct broadcasting to (958, 4096)

        kmm_acc = self.knn_evaluate(Xs_kmm, Ys, Xt, Yt)
        return tca_acc, kmm_acc, Xs_tca, Xt_tca, Xs_kmm

    def visualize_all(self, Xs_orig, Xt_orig, Xs_tca, Xt_tca, Xs_kmm, Ys, Yt):
        """Three-view comparative visualization"""
        # Original data distribution
        self.plot_distribution(self.ax1, Xs_orig, Xt_orig, Ys, Yt, "Original Data Distribution")

        # TCA transformed distribution
        self.plot_distribution(self.ax2, Xs_tca, Xt_tca, Ys, Yt, "TCA Transformation Result")

        # KMM weighted distribution
        self.plot_kmm_distribution(self.ax3, Xs_kmm, Xt_orig, Ys, Yt)

        self.canvas.draw()

    def plot_distribution(self, ax, Xs, Xt, Ys, Yt, title):
        """Generic distribution plotting"""
        ax.clear()
        X_combined = np.vstack([Xs, Xt])
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_combined)

        # Plot source and target domains
        scatter1 = ax.scatter(X_tsne[:len(Xs), 0], X_tsne[:len(Xs), 1],
                              c=Ys, cmap='tab10', s=30, alpha=0.8, label='Source')
        scatter2 = ax.scatter(X_tsne[len(Xs):, 0], X_tsne[len(Xs):, 1],
                              c=Yt, cmap='tab10', marker='x', s=30, alpha=0.8, label='Target')

        ax.set_title(title)
        ax.legend(handles=[scatter1, scatter2])

    def plot_kmm_distribution(self, ax, Xs, Xt, Ys, Yt):
        """KMM weighted visualization"""
        ax.clear()
        # Normalize weights
        weights = (self.beta - self.beta.min()) / (self.beta.max() - self.beta.min()) * 150 + 10

        X_combined = np.vstack([Xs, Xt])
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_combined)

        # Plot weighted source domain
        scatter1 = ax.scatter(X_tsne[:len(Xs), 0], X_tsne[:len(Xs), 1],
                              c=Ys, cmap='tab10', s=weights, alpha=0.7, label='Weighted Source')
        scatter2 = ax.scatter(X_tsne[len(Xs):, 0], X_tsne[len(Xs):, 1],
                              c=Yt, cmap='tab10', marker='x', s=30, alpha=0.7, label='Target')

        ax.set_title("KMM Sample Weight Distribution\n(Point size indicates weight)")
        ax.legend(handles=[scatter1, scatter2])

    def knn_evaluate(self, Xs, Ys, Xt, Yt):
        """KNN evaluation"""
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(Xs, Ys.ravel())
        return model.score(Xt, Yt.ravel())

    def save_results(self):
        """Save visualization results"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            self.fig.savefig(filename, bbox_inches='tight')

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


class KMM:
    """KMM实现（基于用户提供的代码优化）"""

    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps is None:
            self.eps = self.B / np.sqrt(ns)

        # 计算核矩阵
        K = self._kernel(Xs, Xs)
        kappa = np.sum(self._kernel(Xs, Xt) * float(ns) / float(nt), axis=1)

        # 转换为cvxopt格式
        Q = matrix(K.astype(np.double))
        p = matrix(-kappa.astype(np.double))
        G = matrix(np.vstack([np.ones((1, ns)),
                              -np.ones((1, ns)),
                              np.eye(ns),
                              -np.eye(ns)]))
        h = matrix([ns * (1 + self.eps),
                    ns * (self.eps - 1)] +
                   [self.B] * ns +
                   [0.0] * ns)

        # 求解二次规划
        solvers.options['show_progress'] = False
        solution = solvers.qp(Q, p, G, h)
        return np.array(solution['x']).ravel()

    def _kernel(self, X1, X2=None):
        """核函数计算"""
        if self.kernel_type == 'linear':
            return sklearn.metrics.pairwise.linear_kernel(X1, X2)
        elif self.kernel_type == 'rbf':
            return sklearn.metrics.pairwise.rbf_kernel(X1, X2, gamma=self.gamma)
        else:
            raise ValueError("不支持的核类型")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure('Accent.TButton', foreground='white', background='#0078D7')
    app = TCAGUI(root)
    root.mainloop()