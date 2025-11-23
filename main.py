import numpy as np
import sys
import os
import torch
import pygame

from snake_game_ai import SnakeGameAI
from agent import Agent

def train():
    
    toplam_skor = 0
    en_yuksek_skor = 0
    
    game = SnakeGameAI(genislik=800, yukseklik=600)
    agent = Agent()

    print("🤖 Eğitim Başlatılıyor... (Kapatmak için pencereyi kapatın)")

    while True:
        try:
            state_old = game.get_state()
            final_move = agent.get_action(state_old)
            next_state, reward, done, score = game.play_step(np.argmax(final_move))

            agent.train_short_memory(state_old, final_move, reward, next_state, done)
            agent.remember(state_old, final_move, reward, next_state, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > en_yuksek_skor:
                    en_yuksek_skor = score
                    agent.save_model()
                    print(f"💾 Model Kaydedildi! Yeni Rekor: {en_yuksek_skor}")

                print(f'Oyun: {agent.n_games} | Skor: {score} | Rekor: {en_yuksek_skor}')
                
        except pygame.error:
            print("Eğitim sonlandırıldı.")
            sys.exit()

def play_saved_model(model_path='./model/model.pth'):
    
    game = SnakeGameAI(genislik=800, yukseklik=600)
    agent = Agent()
    
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()
        print("✅ Kayıtlı model yüklendi, test başlıyor...")
    else:
        print(f"❌ '{model_path}' bulunamadı. Önce eğitimi çalıştırın (train).")
        return

    agent.epsilon = 0

    while True:
        try:
            state_old = game.get_state()
            final_move = agent.get_action(state_old)
            _, _, done, score = game.play_step(np.argmax(final_move))
            
            if done:
                print(f'Oyun Bitti! Skor: {score}')
                game.reset()
                
        except pygame.error:
            sys.exit()

if __name__ == '__main__':
    train()
    # play_saved_model()
