from con2img import draw_contourmtcs2image as draw
import sys
import pickle

fn = sys.argv[1]
objects = []
with (open(fn, "rb")) as openfile:
  while True:
      try:
          objects.append(pickle.load(openfile))
      except EOFError:
          break

for d in objects[1:]:
  contourDict = d[1]
  image = d[0]
  cntrs = []
  rgbs = []
  for (mode,contourPoints) in contourDict.items():
      # choose color
      if mode == 'ln':
          rgb = [1, 0, 0]
      elif mode == 'lp':
          rgb = [0, 1, 0]
      elif mode == 'rn':
          rgb = [1, 1, 0]
      else:
          rgb = None
      if rgb is not None:
          cntrs.append(contourPoints)
          rgbs.append(rgb)
  if len(cntrs) > 0:
      draw(image, cntrs, rgbs)
