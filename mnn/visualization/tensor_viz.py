"""mnn.visualization.tensor_viz — Tensor and algebra visualizations."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List

class TensorVisualizer:
    def __init__(self, style="dark_background", figsize=(9,7)):
        self.style=style; self.figsize=figsize; plt.style.use(style)
    def plot_tensor_heatmap(self, tensor, title="Tensor", cmap="RdBu_r", save=None, show=False):
        t=np.array(tensor,dtype=float)
        if t.ndim==1: t=t.reshape(1,-1)
        elif t.ndim>2: t=t.reshape(t.shape[0],-1)
        fig,ax=plt.subplots(figsize=self.figsize)
        im=ax.imshow(t,cmap=cmap,aspect='auto',vmin=-np.abs(t).max(),vmax=np.abs(t).max())
        plt.colorbar(im,ax=ax); ax.set_title(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_cayley_table(self, table, elements, title="Cayley Table", save=None, show=False):
        n=len(elements); fig,ax=plt.subplots(figsize=self.figsize)
        ax.imshow(table,cmap='tab20',aspect='equal')
        ax.set_xticks(range(n)); ax.set_xticklabels([str(e) for e in elements],fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels([str(e) for e in elements],fontsize=7)
        for i in range(n):
            for j in range(n): ax.text(j,i,str(elements[table[i,j]]),ha='center',va='center',fontsize=6,color='white')
        ax.set_title(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_lie_algebra(self, basis, title="Lie Algebra Basis", save=None, show=False):
        d=len(basis); cols=min(4,d); rows=(d+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(cols*3.5,rows*3))
        axes=np.array(axes).flatten()
        for i,(e,ax) in enumerate(zip(basis,axes)):
            er=np.real(e); im=ax.imshow(er,cmap='RdBu_r',vmin=-np.abs(er).max(),vmax=np.abs(er).max())
            ax.set_title(f"e_{i}",fontsize=9); plt.colorbar(im,ax=ax)
        for ax in axes[d:]: ax.set_visible(False)
        plt.suptitle(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_theta_function(self, theta_fn, z_range=(-1,1), tau=1j, n=300, title="Theta Function", save=None, show=False):
        zs=np.linspace(*z_range,n); vals=np.array([theta_fn(z,tau) for z in zs])
        fig,axes=plt.subplots(2,1,figsize=(10,7))
        axes[0].plot(zs,np.real(vals),color='cyan',lw=1.2,label="Re"); axes[0].plot(zs,np.imag(vals),color='orange',lw=1.2,label="Im")
        axes[0].legend(); axes[0].set_title(title)
        axes[1].plot(zs,np.abs(vals),color='lime',lw=1.2,label="|θ|"); axes[1].legend()
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def save(self,fig,path,dpi=200): fig.savefig(path,dpi=dpi,bbox_inches='tight')
