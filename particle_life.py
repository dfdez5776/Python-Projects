import pygame 
import numpy as np
#from numba import jit
import os

#i fixed this
class ParticleLife():

    def __init__(self, n_types, n_particles, K, friction):
        self.n_types = n_types
        self.n_half1 = int(5/2)
        self.n_half2 = int(self.n_types/2) + 5 % 2
        self.colors =  360 // self.n_types 
        self.n_particles = n_particles
        self.K = K
        self.friction = friction
        self.width = 300  ##not sure how big to make this
        self.height = 300

        self.mask = np.concatenate((np.ones((self.n_types, self.n_half1))*-1, np.ones((self.n_types, self.n_half2))), axis = 1)

        self.forces =  np.random.randint(3, 10, (self.n_types, self.n_types))/10*self.mask
        self.minDistance = np.random.randint(30, 50, (self.n_types, self.n_types))
        self.radius = np.random.randint(70, 250, (self.n_types, self.n_types))


        self.positions = np.random.rand(self.n_particles, 2) * [self.width, self.height]
        self.velocities = np.zeros((self.n_particles, 2))
        self.types = np.random.randint(self.n_types, size = self.n_particles)  
        
        

    def update_params(self):
        new_positions = np.empty_like(self.positions)
        new_velocities = np.empty_like(self.velocities)

        for i in range(self.n_particles):
            total_force_x, total_force_y = 0.0, 0.0
            i_pos_x, i_pos_y = self.positions[i]
            i_vel_x, i_vel_y = self.velocities[i]
            i_type = self.types[i]


            for j in range(self.n_particles):
                if i == j:
                    continue
                
                #Direction
                j_pos_x, j_pos_y = self.positions[j]
                j_type = self.types[j]

                j_dir_x = j_pos_x - i_pos_x
                j_dir_y = j_pos_y - i_pos_y

                if j_dir_x > 0.5*self.height:
                    j_dir_x -= self.width
                if j_dir_x < -0.5*self.width:
                    j_dir_x += self.width
                if j_dir_y > 0.5*self.height:
                    j_dir_y -= self.height
                if j_dir_y < -0.5*self.height:
                    j_dir_y += self.height

                distance = np.sqrt(j_dir_x**2 + j_dir_y**2) 

                if distance > 0 :
                    j_dir_x =  j_dir_x / distance
                    j_dir_y = j_dir_y / distance

                
                if distance < self.minDistance[i_type, j_type]:
                    force = abs(self.forces[i_type, j_type]) * -3 * (1 - distance / self.minDistance[i_type, j_type]) * self.K
                    total_force_x += j_dir_x * force
                    total_force_y += j_dir_y * force
                    
                if distance < self.radius[i_type, j_type]:
                    force = self.forces[i_type, j_type] * (1 - distance / self.radius[i_type, j_type]) * self.K
                    total_force_x += j_dir_x * force
                    total_force_y += j_dir_y * force
                
            new_vel_x = i_vel_x + total_force_x
            new_vel_y = i_vel_y + total_force_y
            new_pos_x = (i_pos_x + new_vel_x) % self.width
            new_pos_y = (i_pos_y + new_vel_y) % self.height
            new_vel_x *= self.friction
            new_vel_y *= self.friction

            new_positions[i] = new_pos_x, new_pos_y
            new_velocities[i] = new_vel_x, new_vel_y
          

        self.positions = new_positions
        self.velocities = new_velocities
            
            

        
        return new_positions, new_velocities
    



    def set_parameters(self):
        print("doing somethings")

    def save_screen(screen):
        if not os.path.exists("screenshots"):
         os.makedirs("screenshots")
      
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d%H%M%S")
        filename = f"screenshots/particles_{time_str}.png"
        pygame.image.save(screen, filename)

def main():


    
    n_types = 5
    n_particles = 50
    K = 0.05
    friction = 0.85

    particle_life = ParticleLife(n_types, n_particles, K, friction)

    #initialize Pygame
    pygame.init()

    screen = pygame.display.set_mode((particle_life.width, particle_life.height))
    pygame.display.set_caption("Particle Swarm Simulation")
    clock = pygame.time.Clock()

    while True:
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        runing = False
        #    elif event.type == pygame.KEYDOWN:
        #        if event.key == pygame.K_r:
        #            forces, min_distances, radii = particle_life.set_parameters()
        #        elif event.key == pygame.K_s:
        #            particle_life.save_screen(screen)
        
        screen.fill((0,0,0))

        positions, velocities = particle_life.update_params()

        for i in range(particle_life.n_particles):
            color = pygame.Color(0)
            color.hsva = (particle_life.types[i] * particle_life.colors, 100, 100, 100)
            pygame.draw.circle(screen, color, (int(positions[i, 0]), int(positions[i, 1])), 2)
            #print("plot", int(positions[i, 0]), int(positions[i, 1]))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


    particle_life.update_params()

if __name__ == "__main__":
    main()
    





#x_pos1 = positions[0,1]
#print(types)
#rint( np.concatenate(np.ones((n_types, n_half1)), np.ones((n_types, n_half2))))