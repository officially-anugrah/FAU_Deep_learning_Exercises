from pattern import *

def main():
    checker = Checker(4, 2)
    checker.draw()
    checker.show()

    circle = Circle(200, 10, (20,80))
    circle.draw()
    circle.show()

    spectrum = Spectrum(256)
    spectrum.draw()
    spectrum.show()

if __name__ == "__main__":
    main()