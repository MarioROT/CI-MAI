from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple
import numpy as np

class custom_grids():
  """"
  Function to print images in a personalized grid of images according to a layout provided
  or a simple arrangment auto calculated according to the number of columns and rows provided
  it is also possible to add layers of effects to some images as squares, lines, etc.
  """
  def __init__(self,
             imgs: List,
             rows: int = 1,
             cols: int = 1,
             titles: List = None,
             suptitle: str = None, 
             order: List = None,
             figsize: Tuple = (10,10),
             axis: str = None,
             cmap: str = None,
             title_size: int = 12,
             use_grid_spec: bool = True,
             show: bool = True,
             upload: bool = None,
             run = None
             ):
      self.imgs = imgs
      self.rows = rows
      self.cols = cols
      self.titles = titles
      self.suptitle = suptitle
      self.order = order
      self.figsize = figsize
      self.axis = axis
      self.cmap = cmap
      self.title_size = title_size
      self.use_grid_spec = use_grid_spec
      self.len_imgs = 0
      self.fig = None
      self.axs = None

      if not self.order:
        self.order = [[i, [j, j + 1]] for i in range(self.rows) for j in range(self.cols)]

      if show:
          self.show()

      if upload:
          run[upload].upload(self.fig)

  def __len__(self):
    return len(self.imgs)

  def show(self): 
    if not self.use_grid_spec:
      self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=self.figsize)
      self.fig.suptitle(self.suptitle)
      if self.rows <= 1 and self.cols <= 1:
        for idx, img in enumerate(self.imgs):
          self.axs.imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs.axis(self.axis)
          if self.titles:
            self.axs.set_title(self.titles[idx], fontsize=self.title_size)
          self.len_imgs += 1
      elif self.rows <= 1 or self.cols <= 1:
        for idx, img in enumerate(self.imgs):
          self.axs[idx].imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs[idx].axis(self.axis)
          if self.titles:
            self.axs[idx].set_title(self.titles[idx], fontsize= self.title_size)
          self.len_imgs += 1
      else:
        for idx, img in enumerate(self.imgs):
          row = round(np.floor(self.len_imgs/self.cols))
          column = self.len_imgs%self.cols
          self.axs[row][column].imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs[row][column].axis(self.axis)
          if self.titles:
            self.axs[row][column].set_title(self.titles[idx], fontsize= self.title_size)
          self.len_imgs += 1
    else:
      self.fig = plt.figure(constrained_layout=True, figsize=self.figsize)
      self.fig.suptitle(self.suptitle)
      gs = GridSpec(self.rows, self.cols, figure=self.fig)
      for n, (i, j) in enumerate(zip(self.imgs, self.order)):
        im = self.fig.add_subplot(gs[j[0], j[1][0]:j[1][1]])
        if self.cmap:
          im.imshow(i, cmap=self.cmap)
        else:
          im.imshow(i)
        if self.axis:
          im.axis('off')
        if self.titles:
          im.set_title(self.titles[n], fontsize= self.title_size)
    plt.show()

  def add_plot(self, title=None, axis=None, position=None, last=False):
    if self.use_grid_spec:
      warnings.warn("To add graphics you need to set 'use_grid_spec' to false when instantiating the class.")
      return 0
    if self.len_imgs >= (self.rows*self.cols):
      warnings.warn("There is no space available to add a plot. Adjust the number of rows and columns when instantiating the class.")
      return 0
    if not position:
      nextr = round(np.floor(self.len_imgs/self.cols))
      nextc = self.len_imgs%self.cols
      position = [nextr, nextc]

    self.len_imgs += 1

    if last and (self.len_imgs < (self.rows*self.cols)):
      for e in range((self.rows*self.cols)-self.len_imgs):
        nextr = round(np.floor(self.len_imgs/self.cols))
        nextc = self.len_imgs%self.cols
        self.axs[nextr][nextc].axis("off")
        self.len_imgs += 1

    if self.rows <= 1:
      if self.cols <= 1:
        if axis:
          self.axs.axis(axis)
        if title:
          self.axs.set_title(title, fontsize= self.title_size)
        return self.axs
      else:
        if axis:
          self.axs[position[1]].axis(axis)
        if title:
          self.axs[position[1]].set_title(title, fontsize= self.title_size)
        return self.axs[position[1]]
    else:
      if axis:
        self.axs[position[0]][position[1]].axis(axis)
      if title:
        self.axs[position[0]][position[1]].set_title(title, fontsize= self.title_size)
      return self.axs[position[0]][position[1]]

def display_grid(xs, titles, rows, cols, figsize=(20,20), upload=False):
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            i = r * rows + c
            ax[r, c].imshow(xs[i], cmap='Set3')
            ax[r, c].set_title(titles[i])
            ax[r, c].set_xticklabels([])
            ax[r, c].set_yticklabels([])
    fig.tight_layout()
    # plt.show()
    if upload:
        run[upload].upload(fig)

