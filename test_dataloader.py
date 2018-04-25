from parsing import parse_dicom_file, parse_contour_file, poly_to_mask
import numpy as np
from dataset import DICOMDataset, SkipSampleError
from dataloader import DataLoader
import pytest


def test_parse_contour_file(tmpdir):
    path = tmpdir.join('contours.txt')
    points = [(0.5, 39.5), (123.4, 45.2), (0.4, 76.0)]
    np.savetxt(str(path), points)
    contour = parse_contour_file(str(path))
    assert len(points) == len(contour)
    for p1, p2 in zip(points, contour):
        assert p1 == p2


def test_poly_to_mask():
    width = 10
    height = 6
    # create rectangular contour
    contour = [(1.5, 1.5), (8.5, 1.5), (8.5, 4.5), (1.5, 4.5)]
    true_mask = np.zeros((height, width), dtype=bool)
    true_mask[2:4, 2:8] = True

    mask = poly_to_mask(contour, width=width, height=height)
    assert(mask.shape[0] == height)
    assert(mask.shape[1] == width)
    print(true_mask)
    print(mask)
    assert (true_mask == mask).all()

    # check that empty contour create zero mask
    mask = poly_to_mask([], width=width, height=height)
    assert ((mask == False).all())


def test_dataset(tmpdir):
    # create fake dataset
    tmpdir.join('link.csv').write('patient_id,original_id\nSCD0000101,SC-HF-I-1\n')
    icontours_path = tmpdir.mkdir('contourfiles').mkdir('SC-HF-I-1').mkdir('i-contours')
    icontours_path.join('IM-0001-0002-icontour-manual.txt').write('2. 1.\n20. 30.\n 10. 10.')
    icontours_path.join('IM-0001-0022-icontour-manual.txt').write('22. 1.\n20. 30.\n 10. 10.')
    icontours_path.join('IM-0001-0042-icontour-manual.txt').write('42. 1.\n20. 30.\n 10. 10.')
    icontours_path.join('IM-0001-0099-icontour-manual.txt').write('99. 1.\n20. 30.\n 10. 10.')
    ocontours_path = tmpdir.join('contourfiles').join('SC-HF-I-1').mkdir('o-contours')
    ocontours_path.join('IM-0001-0002-ocontour-manual.txt').write('2. 1.\n20. 30.\n 10. 10.')
    ocontours_path.join('IM-0001-0022-ocontour-manual.txt').write('22. 1.\n20. 30.\n 10. 10.')
    ocontours_path.join('IM-0001-0042-ocontour-manual.txt').write('42. 1.\n20. 30.\n 10. 10.')
    dicoms_path = tmpdir.mkdir('dicoms').mkdir('SCD0000101')
    dicoms_path.join('1.dcm').write('1')
    dicoms_path.join('2.dcm').write('2')
    dicoms_path.join('42.dcm').write('42')
    dicoms_path.join('99.dcm').write('99')

    dataset = DICOMDataset(str(tmpdir), only_complete=True)
    assert len(dataset) == 2
    with pytest.raises(SkipSampleError):
        dataset[0]

    dataset = DICOMDataset(str(tmpdir), only_complete=False)
    assert len(dataset) == 3


def test_dataloader():
    data_path = 'data/final_data'
    data = DICOMDataset(data_path)
    batch_size = 4
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    for ib, batch in enumerate(loader):
        for i in range(batch['pixel_data'].shape[0]):
            assert (batch['pixel_data'][i, :, :] == data[ib*batch_size + i]['pixel_data']).all()
            assert (batch['imask'][i, :, :] == data[ib*batch_size + i]['imask']).all()
            assert (batch['omask'][i, :, :] == data[ib*batch_size + i]['omask']).all()
