import cv2
import numpy as np

from main import find_ball_by_color


def test_orange_ball_detection():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 26, (0, 140, 255), -1)

    detection = find_ball_by_color(img)
    assert detection is not None, 'Expected an orange ball to be detected'
    x, y, conf = detection
    assert abs(x - 150) < 10, f'Center x wrong: {x}'
    assert abs(y - 150) < 10, f'Center y wrong: {y}'
    assert conf > 0.2, f'Confidence too low: {conf}'


def main():
    test_orange_ball_detection()
    print('ball detection fallback test passed')


if __name__ == '__main__':
    main()
