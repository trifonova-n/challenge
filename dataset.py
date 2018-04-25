from parsing import parse_contour_file, parse_dicom_file, poly_to_mask
from pathlib import Path


class SkipSampleError(Exception):
    pass


class DICOMDataset(object):
    """
    Dataset class to access DICOM images and masks.
    Each element is dictionary with fields 'pixel_data' and 'mask'
    :param path: path to directory containing link.csv
    :param transform (callable, optional): if specified, called for each
           sample with fields 'pixel_data', 'icontour' and 'ocontour'
    :param only_complete (bool) if True, load only elements containing both icontour and ocontour
    """
    def __init__(self, path, transform=None, only_complete=True):
        self.transform = transform
        self.only_complete = only_complete
        self.filenames = []
        path = Path(path)
        self.path = path
        self.filenames = self.load_filenames()

    def load_link_list(self):
        with (self.path / 'link.csv').open() as linkfile:
            link_list = [l.strip().split(',') for l in linkfile.readlines()][1:]
        return link_list

    def get_contour_file(self, contour_dir, id, type):
        contours_pattern = 'IM-0001-%04d-%scontour-manual.txt'
        contour_file = self.path / 'contourfiles' / contour_dir / ('%s-contours' % type) / (contours_pattern % (id, type))
        return contour_file

    def load_filenames(self):
        filenames = []
        link_list = self.load_link_list()

        # for each DICOM file save corresponding contour file if exists
        for dicoms_dir, contour_dir in link_list:
            dicom_files = (self.path / 'dicoms' / dicoms_dir).glob('*.dcm')
            for dicom_file in dicom_files:
                id = int(str(dicom_file.stem))
                icontour_file = self.get_contour_file(contour_dir, id, 'i')
                ocontour_file = self.get_contour_file(contour_dir, id, 'o')
                i_exists = icontour_file.exists()
                o_exists = ocontour_file.exists()
                if self.only_complete and i_exists and o_exists:
                    filenames.append((str(dicom_file), str(icontour_file), str(ocontour_file)))
                elif not self.only_complete and i_exists:
                    filenames.append((str(dicom_file), str(icontour_file), ''))
        return filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        :param index (int): sample index
        :return: dictionary with fields 'pixel_data' and 'imask' and 'omask'
        """
        sample = parse_dicom_file(self.filenames[index][0])
        if sample is None:
            raise SkipSampleError()
        if self.filenames[index][1]:
            icontour = parse_contour_file(self.filenames[index][1])
            sample['icontour'] = icontour
        if self.filenames[index][2]:
            ocontour = parse_contour_file(self.filenames[index][2])
            sample['ocontour'] = ocontour
        if self.transform:
            sample = self.transform(sample)
        height = sample['pixel_data'].shape[0]
        width = sample['pixel_data'].shape[1]
        if 'icontour' in sample:
            sample['imask'] = poly_to_mask(sample['icontour'], width=width, height=height)
            del sample['icontour']
        if 'ocontour' in sample:
            sample['omask'] = poly_to_mask(sample['ocontour'], width=width, height=height)
            del sample['ocontour']
        return sample
