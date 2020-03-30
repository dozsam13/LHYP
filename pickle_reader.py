from con2img import draw_contourmtcs2image as draw
import sys
import pickle
from domain.patient_data import PatientData

fn = sys.argv[1]
objects = []
with (open(fn, "rb")) as openfile:
  while True:
      try:
          objects.append(pickle.load(openfile))
      except EOFError:
          break

patient_data = objects[0]
print(type(patient_data.systole))
for systoleDiastole in patient_data.systole + patient_data.diastole:
  contourDict = systoleDiastole[1]
  image = systoleDiastole[0]
  slc_frm = systoleDiastole[2]
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
    draw(image, cntrs, rgbs, str(slc_frm))