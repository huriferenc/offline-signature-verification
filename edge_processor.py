#!/usr/bin/env python
import numpy as np
import cv2
import concurrent.futures

from params import C, EPSILON, SEGMENT, START_SEGMENT


class EdgeProcessor:
    # Freeman’s chain code direction vectors for each of the eight directions
    # -> 0 1 2 3 4 6 7
    freeman_chain_vector_direction = {(0, 1): 0, (-1, 1): 1, (-1, 0): 2, (-1, -1): 3,
                                      (0, -1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7}
    # { 0: (0, 1), 1: (1,1), ...}
    freeman_chain_direction_vector = dict(
        (v, k) for k, v in freeman_chain_vector_direction.items())

    def __init__(self, l, filename, subfolder, savingfolder) -> None:
        """EdgeProcessor's Constructor
        Args:
            l (int): Küszöbérték hossza (length threshold)
            Egy quasi-egyenes vonalszakasz csak akkor kerül figyelembevételre egy osztályban,
            ha annak hossza legalább előre meghatározott küszöbérték (l).
            "We experimented with various l values empirically and observed that
            the system works well for l = 4."

            filename (str): alairas kep fajl neve
            subfolder (str): az alairas kepet tartalmazo mappa
            savingfolder (str): a kvazi-egyenes szakaszokat tartalmazo kep mentesi helye
        """

        self.l = l

        self.filename = filename

        # Mentési hely inicializálása
        self.savingpath = savingfolder

        # Mentési hely létrehozása
        self.savingpath.mkdir(parents=True, exist_ok=True)

        self.Q_array = []

        # Betöltjük a képet
        img_origin = cv2.imread(str((subfolder / self.filename).resolve()))

        # Kép szürkeárnyalatosra alakítása
        img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

        # Kép simítása Gauss szűrővel
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Canny éldetekció
        self.canny_edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # Határvonal pixelek (boundary pixels)
        E_y, E_x = np.where(self.canny_edges != 0)
        self._E = [[E_y[i], E_x[i]]
                   for i in range(0, len(E_y))]  # [(123, 23), (56, 1), ...]

        self._P = len(self._E)

        # Sizes
        self.height, self.width = self.canny_edges.shape[:2]

    @property
    def E(self):
        return self._E

    @property
    def class_segments(self):
        return self.Q_array

    def dir(self, p1: tuple, p2: tuple):
        dy: int = p2[0] - p1[0]
        dx: int = p2[1] - p1[1]

        if (dy, dx) in self.freeman_chain_vector_direction:
            return self.freeman_chain_vector_direction[(dy, dx)]

        return None

    def is_visited(self, p: tuple, Visited: list):
        return p in Visited

    def get_unvisited_neighbors(self, p: tuple, Visited: list):
        neighbors = []

        p_y, p_x = p

        for [dy, dx] in self.freeman_chain_vector_direction:
            neighbour = [p_y + dy, p_x + dx]
            if neighbour in self._E and neighbour not in Visited:
                neighbors.append(neighbour)

        return neighbors

    def get_unvisited_neighbour_by_dir(self, p: tuple, dir: int, Visited: list):
        [dy, dx] = self.freeman_chain_direction_vector[dir]
        neighbors = self.get_unvisited_neighbors(p, Visited)

        if len(neighbors) == 0:
            return None

        p_y, p_x = p
        neighbour = [p_y + dy, p_x + dx]

        if neighbour in neighbors:
            return neighbour

        return None

    def get_two_unvisited_sibling_pixels_by_n(self, start: tuple, n: int, Visited: list):
        index = 0

        if start is not None:
            index = self._E.index(start)

        for i in range(index, len(self._E)-1):
            p1 = self._E[i]

            if self.is_visited(p1, Visited):
                continue

            p2 = self.get_unvisited_neighbour_by_dir(p1, n, Visited)

            if p2 is not None:
                return [p1, p2]

        return [start, None]

    ################################################################
    # Procedure Extend-Segment
    ################################################################
    def extend_segment(self, p: tuple, n: int, s: int, qi: list, Visited: list):
        global EPSILON

        if s == EPSILON:
            while True:
                p_n = self.get_unvisited_neighbour_by_dir(p, n, Visited)

                if p_n is None:
                    return

                qi.append(p_n)
                Visited.append(p_n)
                p = p_n
        else:
            d = n
            while True:
                p_n = self.get_unvisited_neighbour_by_dir(p, n, Visited)
                p_s = None if s == EPSILON else self.get_unvisited_neighbour_by_dir(
                    p, s, Visited)

                if p_n is None and p_s is None:
                    return

                if d == n:
                    if p_n is not None:
                        qi.append(p_n)
                        Visited.append(p_n)
                        p = p_n
                        d = n
                        continue
                    elif p_s is not None:
                        qi.append(p_s)
                        Visited.append(p_s)
                        p = p_s
                        d = s
                        continue
                    else:
                        return
                elif d == s:
                    if p_n is not None:
                        qi.append(p_n)
                        Visited.append(p_n)
                        p = p_n
                        d = n
                        continue
                    else:
                        return
                else:
                    return

    ################################################################
    # Detect-Quasi-Straight-Segments algoritmus
    ################################################################
    def DQSS(self, n: int, s: int, l=4):
        global EPSILON

        ################################################################
        # Az i-edik osztályban lévő él szegmenseinek száma
        ni = 0
        # Az E élhalmazban lévő, az i-edik osztályba tartozó él pixeleinek száma
        pi = 0
        ################################################################

        # [ [(23, 32), (56, 9), ...], [(1, 3), (33, 45), ...], [...], ... ]
        Q = []
        # [(23, 32), (56, 9), ...]
        qi = []
        # [(23, 32), (56, 9), ...]
        Visited = []

        p1 = None
        p2 = None

        [p1, p2] = self.get_two_unvisited_sibling_pixels_by_n(p1, n, Visited)
        if p2 is None:
            return [[], 0, 0]

        qi = [p1, p2]
        Visited = [p1, p2]

        while True:
            self.extend_segment(p2, n, s, qi, Visited)
            self.extend_segment(p1, (n + 4) %
                                8, ((s + 4) % 8 if s != EPSILON else EPSILON), qi, Visited)

            if len(qi) >= l:
                ni = ni + 1
                pi = pi + len(qi)

                Q.append(qi)
                qi = []

            [p1, p2] = self.get_two_unvisited_sibling_pixels_by_n(
                p2, n, Visited)

            if p2 is None:
                return [Q, ni, pi]

            qi = [p1, p2]
            Visited.append(p1)
            Visited.append(p2)

        return [Q, ni, pi]

    def create_img_from_dqss(self, Q):
        global START_SEGMENT, SEGMENT

        quasi_straight_line_segments = np.zeros(
            (self.height, self.width), dtype=np.uint8)

        for qi in Q:
            for p in qi:
                p_y, p_x = p
                quasi_straight_line_segments[p_y][p_x] = SEGMENT

            quasi_straight_line_segments[qi[0][0]][qi[0][1]] = START_SEGMENT

        return quasi_straight_line_segments

    def export_quasi_straight_line_segments(self, params):
        Q_img, class_name, l = params

        self.savingpath = self.savingpath / \
            f'C_{class_name}_L_{l}_{self.filename}'

        cv2.imwrite(str(self.savingpath.resolve()), Q_img)

    def get_quasi_straight_line_segments_of_class(self, C):
        class_name, c = C

        Q, ni, pi = self.DQSS(**c, l=self.l)

        Q_img = self.create_img_from_dqss(Q)
        self.export_quasi_straight_line_segments([Q_img, class_name, self.l])

        return [Q, ni, pi, Q_img]

    def start(self):
        global C

        DQSS_arr = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            DQSS_arr = executor.map(
                self.get_quasi_straight_line_segments_of_class, C.items())

        for DQSS in DQSS_arr:
            self.Q_array.append(DQSS)
