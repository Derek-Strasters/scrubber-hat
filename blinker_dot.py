from math import floor
from math import pi
from math import sin
from math import cos
from math import sqrt
from time import sleep
from time import monotonic

from sense_hat import SenseHat


class DeltaTimer:
    def __init__(self, **kwargs):
        self.prev_time = 0
        self.pres_time = 0
        target = kwargs.get("target_time", float(1))
        if isinstance(target, float) and target > 0:
            self.target = target
        else:
            raise ValueError("Parameter 'target_time' must be a positive float")

    def set_target_time(self, target_time):
        self.target = target_time

    def start(self):
        self.prev_time = monotonic()

    def restart(self):
        self.start()

    def delta(self):
        """Returns the time since delta was last checked or timer was restarted"""
        pres_time = monotonic()
        ret = pres_time - self.prev_time
        self.prev_time = pres_time
        return ret

    def deltarget(self, **kwargs):
        positive_only = kwargs.get("positive_only", False)
        positive_only = kwargs.get("p", positive_only)
        time_to_targ = self.target - self.delta()
        return time_to_targ if not positive_only else time_to_targ if time_to_targ > 0 else 0

    def target_sleep(self):
        sleep(self.deltarget(p=1))


def hemisphere_force(s_to_cen):
    return s_to_cen

class Bowl:
    def __init__(self, x1, y1, x2, y2, gravity=9.80665, force_map = hemisphere_force):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.gravity = gravity

        self.x_r = (x2 - x1) / 2
        self.y_r = (y2 - y1) / 2
        self.cen = (self.x_r + x1, self.y_r + y1)

        self.forcer = force_map

    def get_force(self, point):
        d_cen = tuple(point[dim] - self.cen[dim] for dim in range(2))

        xforce = g * sin + F * cos - TODO

        return tuple()

def get_accelerometer_mpss(ahat):
    """Test"""
    accel_in_g = ahat.get_accelerometer_raw()
    return {'x': accel_in_g.get('x', 0) * 9.8066, 'y': accel_in_g.get('y', 0) * 9.8066,
            'z': accel_in_g.get('z', 0) * 9.8066}


def usin(partial_unit):
    return 0.5 + sin(2 * pi * partial_unit) / 2


def husin(partial_hunit: float) -> float:
    return usin(partial_hunit / 2 + 0.75)


def ucos(partial_unit):
    return 0.5 + cos(2 * pi * partial_unit) / 2


def to_byte(percent):
    return int(255 * percent)


def split_br(pix):
    p_x0 = int(pix[0])
    p_y0 = int(pix[1])
    p_x1 = p_x0 + 1
    p_y1 = p_y0 + 1

    r_x1 = pix[0] - p_x0
    r_y1 = pix[1] - p_y0
    r_x0 = 1 - r_x1
    r_y0 = 1 - r_y1

    def f(inp):
        return sqrt(inp)

    pix_kernel = [(p_x0, p_y0), (p_x1, p_y0),
                  (p_x0, p_y1), (p_x1, p_y1)]

    fac_kernel = [to_byte(f(r_x0 * r_y0)), to_byte(f(r_x1 * r_y0)),
                  to_byte(f(r_x0 * r_y1)), to_byte(f(r_x1 * r_y1))]



    # def make_white_tup(factor: float) -> tuple:
    #     ret_part = 255 * facfun(factor)
    #     ret = (int(ret_part), int(ret_part), int(ret_part))
    #     return ret

    return tuple(zip(pix_kernel, fac_kernel))


def aa_set_pix(ahat, pix):
    split = split_br(pix)
    for pixel in split:
        if all(7 >= c >= 0 for c in pixel):
            ahat.set_pixel(*pixel)


def aa_set_pixels(ahat, pixels):
    for pixel in pixels:
        aa_set_pix(ahat, pixel)


if __name__ == "__main__":
    hat = SenseHat()
    sleeper = DeltaTimer(target_time=0.003)

    while True:
        hat.clear()
        steps = 750
        for i in range(0, steps):
            funity = i / steps
            apix = (usin(funity) * 7, ucos(funity) * 7)
            aa_set_pix(hat, apix)
            sleeper.target_sleep()
