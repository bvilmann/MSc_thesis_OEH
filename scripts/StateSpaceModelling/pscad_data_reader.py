import numpy as np
import xml.etree.ElementTree as ET
import os

import numpy as np
import xml.etree.ElementTree as ET
import os

class PowerSourceDataLoader:
    def __init__(self, base_filename, directory='.'):
        self.base_filename = base_filename
        self.directory = directory
        self.infx_filename = os.path.join(directory, f"{base_filename}.infx")

    def _get_info_from_infx(self, measurement_name):
        tree = ET.parse(self.infx_filename)
        root = tree.getroot()

        for analog in root.findall(".//Analog"):
            if analog.get("name") == measurement_name:
                return int(analog.get("index")), int(analog.get("dim"))
        return None, None

    def _load_out_file(self, index):
        file_counter = index // 10 + 1
        out_filename = os.path.join(self.directory, f"{self.base_filename}_{file_counter:02d}.out")
        data = np.loadtxt(out_filename, skiprows=0)  # Assuming the first row is a header
        return data

    def get_measurement_data(self, measurement_name,t0=0,t1=9999999):
        index, dim = self._get_info_from_infx(measurement_name)
        if index is None or dim is None:
            raise ValueError(f"Measurement {measurement_name} not found in {self.infx_filename}")

        # Load the first file
        time_series = self._load_out_file(index)[:, 0]
        if index % 10 == 0:
            measurement_data = [self._load_out_file(index-1)[:, 10]]
        else:
            measurement_data = [self._load_out_file(index)[:, index % 10]]

        # If dim > 1, load additional files
        for i in range(1, dim):
            if (index+i) % 10 == 0:
                measurement_data.append(self._load_out_file(index + i - 1)[:, 10])
            else:
                measurement_data.append(self._load_out_file(index + i)[:, (index + i) % 10])

        vals = np.vstack(measurement_data).T
        mask = (time_series >= t0) & (time_series <= t1)
        time_series = time_series[mask]
        time_series -= t0
        vals = vals[mask,:]

        return time_series, vals