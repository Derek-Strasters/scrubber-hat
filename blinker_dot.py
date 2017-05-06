import itertools
import threading
from math import pi
from math import sin
from math import cos
from math import sqrt
from time import sleep
from time import monotonic
from abc import ABCMeta, abstractmethod

from sense_hat import SenseHat


def usin(partial_unit):
    return 0.5 + sin(2 * pi * partial_unit) / 2


def husin(partial_hunit: float) -> float:
    return usin(partial_hunit / 2 + 0.75)


def ucos(partial_unit):
    return 0.5 + cos(2 * pi * partial_unit) / 2


def to_byte_int(percent):
    return int(255 * percent)


class DeltaTimer:
    def __init__(self, target_time=1.0):
        self.prev_time = 0
        self.pres_time = 0
        self.target_time = target_time

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

    def deltarget(self, positive_only=False):
        time_to_targ = self.target_time - self.delta()
        return time_to_targ if not positive_only else time_to_targ if time_to_targ > 0 else 0

    def sleep_target(self):
        sleep(self.deltarget(positive_only=True))


class PhysObject(metaclass=ABCMeta):
    def __init__(self, x=0, y=0):
        self.pos = [x, y]
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.vel = [0, 0]
        self.xv = self.vel[0]
        self.yv = self.vel[1]

    @abstractmethod
    def update(self, acc: tuple, dt):
        pass


class PixelPrintable(metaclass=ABCMeta):
    @abstractmethod
    def print_pixels(self) -> tuple:
        pass


class Bowl(PhysObject):
    def __init__(self, xmin, ymin, xmax, ymax, gravity=9.80665, a_hat=SenseHat()):
        super().__init__()
        self.xtents = xmax, ymax, xmin, ymin
        self.gravity = gravity
        self.hat = a_hat
        self.acc = {'x': 0, 'y': 0, 'z': 0}

        self.phys_objects = []

        self.cen = ((xmax + xmin) / 2, (ymax + ymin) / 2)
        self.radii = xmax - self.cen[0], ymax - self.cen[1]

    def accelerate(self, po: PhysObject) -> tuple:
        a = (self.cartesian_calc_f(self.acc.get('x'), self.radii[0], po.xv, po.x),
             self.cartesian_calc_f(self.acc.get('y'), self.radii[1], po.yv, po.y))
        return a

    def update_accelerometer(self):
        a = self.hat.get_accelerometer_raw()
        g = self.gravity
        self.acc = {'x': a.get('x', 0) * g,
                    'y': a.get('y', 0) * g,
                    'z': a.get('z', 0) * g}

    def cartesian_calc_f(self, a, r, v, x):
        rr = r * r
        xx = x * x
        z = sqrt(rr - xx)
        return v * (z + x) / rr - z * x / rr * self.effective_g() + z / r * a

    def effective_g(self):
        return self.acc.get('z')

    def add_phys_obj(self, phys_obj: PhysObject):
        self.phys_objects.append(phys_obj)

    def update(self, acc: tuple, dt):
        self.update_accelerometer()
        for obj in self.phys_objects:
            obj.update(self.accelerate(obj)[0], self.accelerate(obj)[1], dt)


class Point(PixelPrintable, PhysObject):
    def __init__(self, x=0, y=0):
        super().__init__(x=x, y=y)

    def print_pixels(self):
        def f(x): sqrt(x)

        return aa_kernel(self.pos, f)

    def update(self, acc, dt):
        self.pos += self.vel * dt
        self.vel += acc * dt


class Pixel:
    def __init__(self, i, j, val):
        self.i = i
        self.j = j
        self.val = int(255 * val)

    def add(self, pixel):
        self.val = 255 if self.val + pixel.val > 255 else self.val + pixel.val

    def reset(self):
        self.val = 0


class PixMatrix:
    def __init__(self, cols, rows):
        self.pixel_matrix = [[Pixel(col, row, 0) for col in cols] for row in rows]

    def reset(self):
        for pix in self.single_list():
            pix.resest()

    def set_pixels(self, pixels):
        for pixel in pixels:
            self.pixel_matrix[pixel.i][pixel.j].add(pixel)
            if self.pixel_matrix[pixel.i][pixel.j].val > 255: self.pixel_matrix[pixel.i][pixel.j].val = 255

    def single_list(self):
        return itertools.chain.from_iterable(self.pixel_matrix)


class PixelPrinter(threading.Thread):
    def __init__(self, printable_items, delta_t=0.033):
        super().__init__()
        if all(isinstance(item, PixelPrintable) for item in printable_items):
            self.printable_items = printable_items
        else:
            raise TypeError("All items must be PixelPrintable")
        self.pix_matrix = PixMatrix(7, 7)  # TODO: put these globals somewhere better
        self.delta_timer = DeltaTimer(delta_t)
        self.hat = SenseHat()
        self.running = True

    def run(self):
        timer = self.delta_timer
        timer.start()
        while self.running:
            for item in self.printable_items:
                self.pix_matrix.set_pixels(item.print_pixels())
                self.hat.set_pixels()  # TODO: put stuff here
                self.pix_matrix.reset()
                timer.sleep_target()


def aa_kernel(pos, modifier_func):
    p_00 = int(pos[0]), int(pos[1])
    p_01 = p_00[0] + 1, p_00[1]
    p_10 = p_00[0], p_00[1] + 1
    p_11 = p_01[0], p_10[1]

    ry_1 = pos[1] - p_00[1]
    ry_0 = 1 - ry_1

    rx_1 = pos[0] - p_00[0]
    rx_0 = 1 - rx_1

    kernel_coord = (p_00, p_01,
                    p_10, p_11)

    pix_fraction = (rx_0 * ry_0, rx_1 * ry_0,
                    rx_0 * ry_1, rx_1 * ry_1)

    ret_pixels = map(lambda coord, val: Pixel(coord[0], coord[1], modifier_func(val)), kernel_coord, pix_fraction)
    return ret_pixels


# def aa_set_pix(ahat, pix):
#     split = aa_kernel(pix)
#     for pixel in split:
#         if all(7 >= c >= 0 for c in pixel):
#             ahat.set_pixel(*pixel)
#
#
# def aa_set_pixels(ahat, pixels):
#     for pixel in pixels:
#         aa_set_pix(ahat, pixel)


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
            sleeper.sleep_target()
