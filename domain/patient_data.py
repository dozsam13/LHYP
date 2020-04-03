


class PatientData:

    def __init__(self, scan_id, pathology, volume_data, systole, diastole):
    	self.scan_id = scan_id
    	self.pathology = pathology
    	self.volume_data = volume_data
    	self.systole = systole
    	self.diastole = diastole
    	
    def mid_systole_contours(self):
    	return self.systole[1][1]

    def mid_diastole_contours(self):
    	return self.diastole[1][1]