"""mnn.visualization.geo_viz — Geometry and topology visualizations."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional

class GeometryVisualizer:
    def __init__(self, style="dark_background", figsize=(10,8)):
        self.style=style; self.figsize=figsize; plt.style.use(style)
    def plot_sphere(self, radius=1., center=None, res=50, geodesics=None, title="S²", save=None, show=False):
        c=center or [0,0,0]; u=np.linspace(0,2*np.pi,res); v=np.linspace(0,np.pi,res)
        x=c[0]+radius*np.outer(np.cos(u),np.sin(v)); y=c[1]+radius*np.outer(np.sin(u),np.sin(v))
        z=c[2]+radius*np.outer(np.ones(res),np.cos(v))
        fig=plt.figure(figsize=self.figsize); ax=fig.add_subplot(111,projection='3d')
        ax.plot_surface(x,y,z,alpha=0.2,color='cyan',linewidth=0)
        ax.plot_wireframe(x,y,z,alpha=0.07,color='white',linewidth=0.3)
        if geodesics:
            for g in geodesics: ax.plot(g[:,0],g[:,1],g[:,2],lw=2,color='yellow')
        ax.set_title(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_torus(self, R=2., r=0.8, res=60, title="T²", save=None, show=False):
        u=np.linspace(0,2*np.pi,res); v=np.linspace(0,2*np.pi,res); U,V=np.meshgrid(u,v)
        X=(R+r*np.cos(V))*np.cos(U); Y=(R+r*np.cos(V))*np.sin(U); Z=r*np.sin(V)
        fig=plt.figure(figsize=self.figsize); ax=fig.add_subplot(111,projection='3d')
        ax.plot_surface(X,Y,Z,cmap='plasma',alpha=0.8,linewidth=0); ax.set_title(title); plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def plot_manifold_chart(self, points, values=None, title="Manifold", save=None, show=False):
        if points.shape[1]==2:
            fig,ax=plt.subplots(figsize=self.figsize)
            sc=ax.scatter(points[:,0],points[:,1],c=values,cmap='viridis',s=5) if values is not None else ax.scatter(points[:,0],points[:,1],s=2,color='cyan',alpha=0.6)
            if values is not None: plt.colorbar(sc,ax=ax)
            ax.set_title(title)
        else:
            fig=plt.figure(figsize=self.figsize); ax=fig.add_subplot(111,projection='3d')
            sc=ax.scatter(points[:,0],points[:,1],points[:,2],c=values,cmap='plasma',s=3,alpha=0.7) if values is not None else ax.scatter(points[:,0],points[:,1],points[:,2],s=2,color='cyan',alpha=0.5)
            if values is not None: plt.colorbar(sc,ax=ax)
            ax.set_title(title)
        plt.tight_layout()
        if save: fig.savefig(save,dpi=150,bbox_inches='tight')
        if show: plt.show()
        return fig
    def save(self,fig,path,dpi=200): fig.savefig(path,dpi=dpi,bbox_inches='tight')

