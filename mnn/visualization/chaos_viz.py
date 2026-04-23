"""mnn.visualization.chaos_viz — Strange attractor and chaos visualizations."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import List, Tuple, Optional

class ChaosVisualizer:
    def __init__(self, style="dark_background", figsize=(12,9)):
        self.style=style; self.figsize=figsize; plt.style.use(style)
    def plot_attractor_3d(self, traj, title="Strange Attractor", cmap="plasma", lw=0.4, save=None, show=False):
        x,y,z=traj[:,0],traj[:,1],traj[:,2]; n=len(x)
        c=np.linspace(0,1,n-1); cmap_o=plt.cm.get_cmap(cmap)
        segs=np.stack([traj[:-1],traj[1:]],axis=1)
        fig=plt.figure(figsize=self.figsize); ax=fig.add_subplot(111,projection='3d')
        lc=Line3DCollection(segs,colors=[cmap_o(ci) for ci in c],linewidth=lw,alpha=0.7)
        ax.add_collection3d(lc); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max()); ax.set_zlim(z.min(),z.max())
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(title,fontsize=14)
        sm=plt.cm.ScalarMappable(cmap=cmap_o); sm.set_array([]); plt.colorbar(sm,ax=ax,label="time",shrink=0.6)
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_attractor_2d_projections(self, traj, title="Attractor Projections", cmap="inferno", save=None, show=False):
        x,y,z=traj[:,0],traj[:,1],traj[:,2]; c=np.linspace(0,1,len(x))
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for ax,(xi,yi,xl,yl) in zip(axes,[(x,y,"X","Y"),(x,z,"X","Z"),(y,z,"Y","Z")]):
            ax.scatter(xi,yi,c=c,cmap=cmap,s=0.1,alpha=0.6); ax.set_xlabel(xl); ax.set_ylabel(yl)
        plt.suptitle(title,fontsize=13); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_bifurcation(self, params, values, title="Bifurcation Diagram", xlabel="r", ylabel="x", save=None, show=False):
        fig,ax=plt.subplots(figsize=self.figsize)
        ax.plot(params,values,',',ms=0.3,color='cyan',alpha=0.4)
        ax.set_xlabel(xlabel,fontsize=12); ax.set_ylabel(ylabel,fontsize=12); ax.set_title(title,fontsize=14)
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_poincare_section(self, pts, dims=(0,1), title="Poincaré Section", save=None, show=False):
        if len(pts)==0: return plt.figure()
        fig,ax=plt.subplots(figsize=(8,8))
        ax.scatter(pts[:,dims[0]],pts[:,dims[1]],c=np.arange(len(pts)),cmap='plasma',s=3,alpha=0.8)
        ax.set_xlabel(f"dim {dims[0]}"); ax.set_ylabel(f"dim {dims[1]}"); ax.set_title(title)
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_recurrence(self, R, title="Recurrence Plot", save=None, show=False):
        fig,ax=plt.subplots(figsize=(8,8))
        ax.imshow(R,cmap='binary',origin='lower',interpolation='nearest')
        ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("Time"); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_lyapunov_spectrum(self, spectrum, title="Lyapunov Spectrum", save=None, show=False):
        fig,ax=plt.subplots(figsize=(8,5))
        colors=['red' if l>0 else 'lime' if abs(l)<0.01 else 'cyan' for l in spectrum]
        ax.bar(range(len(spectrum)),spectrum,color=colors,alpha=0.8,edgecolor='white')
        ax.axhline(0,color='white',lw=1.,ls='--'); ax.set_xlabel("Index"); ax.set_ylabel("λ"); ax.set_title(title)
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def save(self,fig,path,dpi=200): fig.savefig(path,dpi=dpi,bbox_inches='tight')
