import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Create the Flappy Bird environment
env = gym.make('FlappyBird-v0', render_mode='human')
state, info = env.reset()
done = False

# Initialize Pygame Keyboard
pygame.init()
screen = pygame.display.get_surface() # Gym has already created a window 

while not done:
    action = 0 # Default-> do nothing, 1-> flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 # Flap

    state, reward, done, truncated, info = env.step(action) 
    env.render()

env.close()
pygame.quit()