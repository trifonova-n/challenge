from parsing import parse_contour_file, parse_dicom_file, poly_to_mask
from pathlib import Path
from pydicom.data import get_testdata_files


class DICOMDataset(object):
    """
    Dataset class to access DICOM images and masks.
    Each element is dictionary with fields 'pixel_data' and 'mask'
    :param path: path to directory containing link.csv
    :param transform (callable, optional): if specified, called for each image, contour pair
    """
    def __init__(self, path, transform=None):
        self.transform = transform
        self.filenames = []
        path = Path(path)
        self.path = path
        self.filenames = self.load_filenames()

    def load_link_list(self):
        with (self.path / 'link.csv').open() as linkfile:
            link_list = [l.strip().split(',') for l in linkfile.readlines()][1:]
        return link_list

    def load_filenames(self):
        contours_pattern = 'IM-0001-%04d-icontour-manual.txt'
        filenames = []
        link_list = self.load_link_list()

        # for each DICOM file save corresponding contour file if exists
        for dicoms_dir, contour_dir in link_list:
            dicom_files = (self.path / 'dicoms' / dicoms_dir).glob('*.dcm')
            for dicom_file in dicom_files:
                id = int(str(dicom_file.stem))
                contour_file = self.path / 'contourfiles' / contour_dir / 'i-contours' / (contours_pattern % id)
                if contour_file.exists():
                    filenames.append((str(dicom_file), str(contour_file)))
        return filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        :param index (int): sample index
        :return: dictionary with fields 'pixel_data' and 'mask'
        """
        sample = parse_dicom_file(self.filenames[index][0])
        contour = parse_contour_file(self.filenames[index][1])
        if self.transform:
            sample['pixel_data'], contour = self.transform(sample['pixel_data'], contour)
        height = sample['pixel_data'].shape[0]
        width = sample['pixel_data'].shape[1]
        mask = poly_to_mask(contour, width=width, height=height)
        sample['mask'] = mask
        return sample
