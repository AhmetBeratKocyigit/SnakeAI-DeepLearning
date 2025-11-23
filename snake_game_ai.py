import pygame
import sys
import random
from collections import namedtuple

Point = namedtuple('Point', 'x, y')

BEYAZ = (255, 255, 255)
KIRMIZI = (200, 0, 0)
MAVI1 = (0, 0, 255)
MAVI2 = (0, 100, 255)
SIYAH = (0, 0, 0)

BLOK_BOYUTU = 20
HIZ = 60  

YONLER = [(1, 0), (0, 1), (-1, 0), (0, -1)]

class SnakeGameAI:

    def __init__(self, genislik=640, yukseklik=480):
        self.genislik = genislik
        self.yukseklik = yukseklik
        
        pygame.init()
        self.font = pygame.font.Font(None, 25)
        self.ekran = pygame.display.set_mode((self.genislik, self.yukseklik))
        pygame.display.set_caption('Yılan Oyunu AI')
        self.saat = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.kafa = Point(self.genislik/2, self.yukseklik/2)
        self.yilan = [self.kafa, 
                      Point(self.kafa.x-BLOK_BOYUTU, self.kafa.y),
                      Point(self.kafa.x-(2*BLOK_BOYUTU), self.kafa.y)]
        
        self.yon = YONLER[0]
        self.skor = 0
        self.elma = None
        self._rastgele_elma_yerlestir()
        self.frame_iteration = 0
        self.engeller = [] 
        return self.get_state()

    def _rastgele_elma_yerlestir(self):
        x = random.randint(0, (self.genislik-BLOK_BOYUTU)//BLOK_BOYUTU)*BLOK_BOYUTU
        y = random.randint(0, (self.yukseklik-BLOK_BOYUTU)//BLOK_BOYUTU)*BLOK_BOYUTU
        self.elma = Point(x, y)
        if self.elma in self.yilan:
            self._rastgele_elma_yerlestir()

    def play_step(self, action):
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        reward = 0
        oyun_sonu = False

        old_dist = abs(self.kafa.x - self.elma.x) + abs(self.kafa.y - self.elma.y)

        self._yon_guncelle(action)
        self._hareket_et(self.yon) 
        self.yilan.insert(0, self.kafa)
        
        if self._carpisti() or self.frame_iteration > 50*len(self.yilan):
            oyun_sonu = True
            reward = -50  
            return self.get_state(), reward, oyun_sonu, self.skor

        if self.kafa == self.elma:
            self.skor += 1
            reward = 10  
            self._rastgele_elma_yerlestir()
        else:
            self.yilan.pop() 
            
            new_dist = abs(self.kafa.x - self.elma.x) + abs(self.kafa.y - self.elma.y)
            
            if new_dist < old_dist:
                reward = 0.1 
            elif new_dist > old_dist:
                reward = -0.5 

        self._oyunu_ciz()
        self.saat.tick(HIZ)
        
        return self.get_state(), reward, oyun_sonu, self.skor

    def _is_danger(self, pt):
        if pt.x > self.genislik - BLOK_BOYUTU or pt.x < 0 or \
           pt.y > self.yukseklik - BLOK_BOYUTU or pt.y < 0:
            return True
        if pt in self.yilan[1:]:
            return True
        return False

    def _carpisti(self, pt=None):
        if pt is None:
            pt = self.kafa
        return self._is_danger(pt)

    def _oyunu_ciz(self):
        self.ekran.fill(SIYAH)
        

        for idx, pt in enumerate(self.yilan):
            renk = MAVI2 if idx == 0 else MAVI1
            pygame.draw.rect(self.ekran, renk, pygame.Rect(pt.x, pt.y, BLOK_BOYUTU, BLOK_BOYUTU))
            
        pygame.draw.rect(self.ekran, KIRMIZI, pygame.Rect(self.elma.x, self.elma.y, BLOK_BOYUTU, BLOK_BOYUTU))
        
        text = self.font.render("Skor: " + str(self.skor), True, BEYAZ)
        self.ekran.blit(text, [0, 0])
        pygame.display.flip()

    def _yon_guncelle(self, action):
        idx = YONLER.index(self.yon)
        if action == 0: 
            yeni_yon = YONLER[idx]
        elif action == 1: 
            yeni_yon = YONLER[(idx + 1) % 4]
        elif action == 2: 
            yeni_yon = YONLER[(idx - 1) % 4]
        self.yon = yeni_yon

    def _hareket_et(self, yon):
        x = self.kafa.x + yon[0] * BLOK_BOYUTU
        y = self.kafa.y + yon[1] * BLOK_BOYUTU
        self.kafa = Point(x, y)

    def get_state(self):
        kafa = self.kafa
        idx = YONLER.index(self.yon)
        
        point_l = Point(kafa.x + self.yon[0]*BLOK_BOYUTU, kafa.y + self.yon[1]*BLOK_BOYUTU)
        yon_r = YONLER[(idx + 1) % 4]
        point_r = Point(kafa.x + yon_r[0]*BLOK_BOYUTU, kafa.y + yon_r[1]*BLOK_BOYUTU)
        yon_u = YONLER[(idx - 1) % 4]
        point_u = Point(kafa.x + yon_u[0]*BLOK_BOYUTU, kafa.y + yon_u[1]*BLOK_BOYUTU)
        
        state = [
            self._is_danger(point_l),
            self._is_danger(point_r),
            self._is_danger(point_u),
            self.yon == YONLER[0],
            self.yon == YONLER[1],
            self.yon == YONLER[2],
            self.yon == YONLER[3],
            self.elma.x < kafa.x, 
            self.elma.x > kafa.x, 
            self.elma.y < kafa.y, 
            self.elma.y > kafa.y
        ]
        return [int(x) for x in state]
