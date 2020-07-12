


class PatientData:
    def __init__(self, scan_id, pathology, volume_data, systole_slices, diastole_slices):
        self.scan_id = scan_id
        self.pathology = pathology
        self.volume_data = volume_data
        self.systole_slices = systole_slices
        self.diastole_slices = diastole_slices