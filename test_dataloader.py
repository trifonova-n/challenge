from parsing import parse_dicom_file, parse_contour_file, poly_to_mask
import numpy as np
from dataset import DICOMDataset, SkipSampleError
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
    contour = [(1.5, 1.5), (8.5, 1.5), (8.5, 4.5), (1.5, 4.5)]
    true_mask = np.zeros((height, width), dtype=bool)
    true_mask[2:4, 2:8] = True
    mask = poly_to_mask(contour, width=width, height=height)
    assert(mask.shape[0] == height)
    assert(mask.shape[1] == width)
    print(true_mask)
    print(mask)
    assert (true_mask == mask).all()

    mask = poly_to_mask([], width=width, height=height)
    assert((mask == False).all())


def test_dataset(tmpdir):
    # create fake dataset
    tmpdir.join('link.csv').write('patient_id,original_id\nSCD0000101,SC-HF-I-1\n')
    contours_path = tmpdir.mkdir('contourfiles').mkdir('SC-HF-I-1').mkdir('i-contours')
    contours_path.join('IM-0001-0002-icontour-manual.txt').write('2. 1.\n20. 30.\n 10. 10.')
    contours_path.join('IM-0001-0022-icontour-manual.txt').write('22. 1.\n20. 30.\n 10. 10.')
    contours_path.join('IM-0001-0042-icontour-manual.txt').write('42. 1.\n20. 30.\n 10. 10.')
    dicoms_path = tmpdir.mkdir('dicoms').mkdir('SCD0000101')
    dicoms_path.join('1.dcm').write('1')
    dicoms_path.join('2.dcm').write('2')
    dicoms_path.join('42.dcm').write('42')

    dataset = DICOMDataset(str(tmpdir))
    assert len(dataset) == 2
    with pytest.raises(SkipSampleError):
        dataset[0]
