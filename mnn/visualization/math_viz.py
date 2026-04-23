"""mnn.visualization.math_viz — Scalar/vector field visualizations."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List, Optional

class MathVisualizer:
    def __init__(self, style="dark_background", figsize=(10,7)):
        self.style=style; self.figsize=figsize; plt.style.use(style)
    def plot_scalar_field_2d(self, f, x_range=(-3,3), y_range=(-3,3), n=300, title="Scalar Field", cmap="viridis", save=None, show=False):
        xs=np.linspace(*x_range,n); ys=np.linspace(*y_range,n); X,Y=np.meshgrid(xs,ys); Z=f(X,Y)
        fig,ax=plt.subplots(figsize=self.figsize)
        im=ax.contourf(X,Y,Z,levels=60,cmap=cmap); ax.contour(X,Y,Z,levels=20,colors='white',alpha=0.3,linewidths=0.5)
        plt.colorbar(im,ax=ax); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_scalar_field_3d(self, f, x_range=(-2,2), y_range=(-2,2), n=80, title="Surface", cmap="plasma", save=None, show=False):
        xs=np.linspace(*x_range,n); ys=np.linspace(*y_range,n); X,Y=np.meshgrid(xs,ys); Z=f(X,Y)
        fig=plt.figure(figsize=self.figsize); ax=fig.add_subplot(111,projection='3d')
        surf=ax.plot_surface(X,Y,Z,cmap=cmap,alpha=0.85,linewidth=0); fig.colorbar(surf,ax=ax,shrink=0.5)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title(title)
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_vector_field_2d(self, F, x_range=(-3,3), y_range=(-3,3), n=20, title="Vector Field", save=None, show=False):
        xs=np.linspace(*x_range,n); ys=np.linspace(*y_range,n); X,Y=np.meshgrid(xs,ys)
        UV=F(X,Y); U,V=UV[0],UV[1]; mag=np.sqrt(U**2+V**2)+1e-15; Un=U/mag; Vn=V/mag
        fig,ax=plt.subplots(figsize=self.figsize)
        q=ax.quiver(X,Y,Un,Vn,mag,cmap='coolwarm',alpha=0.85); plt.colorbar(q,ax=ax,label="|F|")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_streamlines(self, F, x_range=(-3,3), y_range=(-3,3), n=200, title="Streamlines", save=None, show=False):
        xs=np.linspace(*x_range,n); ys=np.linspace(*y_range,n); X,Y=np.meshgrid(xs,ys)
        UV=F(X,Y); U,V=UV[0],UV[1]; speed=np.sqrt(U**2+V**2)
        fig,ax=plt.subplots(figsize=self.figsize)
        strm=ax.streamplot(xs,ys,U,V,color=speed,linewidth=1.5,cmap='inferno',density=1.5)
        plt.colorbar(strm.lines,ax=ax,label="Speed"); ax.set_title(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_fourier_analysis(self, signal, dt=1., title="Fourier Analysis", save=None, show=False):
        n=len(signal); f=np.fft.fftfreq(n,d=dt); sp=np.fft.fft(signal); pos=f>0
        fig,axes=plt.subplots(3,1,figsize=(self.figsize[0],10))
        axes[0].plot(np.arange(n)*dt,signal,color='cyan',lw=0.8); axes[0].set_title("Signal")
        axes[1].plot(f[pos],np.abs(sp[pos]),color='lime',lw=0.8); axes[1].set_title("Amplitude Spectrum")
        axes[2].semilogy(f[pos],np.abs(sp[pos])**2,color='orange',lw=0.8); axes[2].set_title("PSD")
        plt.suptitle(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def save(self,fig,path,dpi=200): fig.savefig(path,dpi=dpi,bbox_inches='tight')
