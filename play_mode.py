import pygame
import sys
import os
import torch
import numpy as np

# Diğer modülleri içe aktar
from snake_game_ai import SnakeGameAI, HIZ
from agent import Agent 

# --- Sabitler ---
BEYAZ = (255, 255, 255)
KIRMIZI = (200, 0, 0)

def display_message(ekran, mesaj, renk, y_offset=0):

    font = pygame.font.Font(None, 75)
    text = font.render(mesaj, True, renk)
    text_rect = text.get_rect(center=(ekran.get_width() / 2, ekran.get_height() / 2 + y_offset))
    ekran.blit(text, text_rect)

def play_with_model(model_path='./model/model.pth'):
    
    game = SnakeGameAI(genislik=800, yukseklik=600)
    agent = Agent()
    
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()  
        print("✅ Eğitilmiş model başarıyla yüklendi. 'R' tuşuna basarak başlatın.")
    else:
        print(f"❌ '{model_path}' bulunamadı. Lütfen önce eğitimi çalıştırın.")
        return


    agent.epsilon = 0
    
    oyun_basladi = False
    oyun_bittimi = False
    son_skor = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    oyun_basladi = True
                    oyun_bittimi = False
                
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        if oyun_basladi and not oyun_bittimi:
            state_old = game.get_state()
            
            final_move = agent.get_action(state_old)
            
            _, _, done, score = game.play_step(np.argmax(final_move))
            
            if done:
                oyun_bittimi = True
                oyun_basladi = False
                son_skor = score

        if not oyun_basladi and not oyun_bittimi:
            game._oyunu_ciz() 
            display_message(game.ekran, "Yapay Zeka Hazir", BEYAZ, -70)
            display_message(game.ekran, "Baslatmak icin 'R' tusuna basin", BEYAZ, 0)
            
        elif oyun_bittimi:
            game._oyunu_ciz() 
            
            display_message(game.ekran, "OYUN BITTI!", KIRMIZI, -70)
            display_message(game.ekran, f"SKOR: {son_skor}", BEYAZ, 0)
            display_message(game.ekran, "Yeniden oynamak icin 'R'", BEYAZ, 70)

        pygame.display.flip()
        if oyun_basladi:
            game.saat.tick(HIZ)
        else:
            game.saat.tick(10)

if __name__ == '__main__':
    play_with_model()
