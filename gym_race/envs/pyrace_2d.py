import pygame
import math

screen_width = 1500
screen_height = 800
check_point = ((1370, 675), (1370, 215), (935, 465), (630, 180), (320, 160), (130, 675), (550, 702)) # race_track_ie.png
"""
Area for text boxes:
750,0 - 1500,100
1055,290 - 1300,605
"""

class Car:
    def __init__(self, car_file, map, pos, surface=None): # map_file
        # self.map = pygame.image.load(map_file)
        self.map = map
        if surface is not None:
            self.surface = surface.copy()
        else:
            self.surface = pygame.image.load(car_file)
            self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = list(pos)
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.check_flag = False
        """
        for d in range(-90, 120, 45): self.check_radar(d)
        for d in range(-90, 105, 15): self.check_radar_for_draw(d)
        """

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars: # or self.radars_for_draw
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def pixel_at(self,x,y):
        try:
            return self.map.get_at((x,y))
        except:
            return (255, 255, 255, 255)

    def check_collision(self, map=None):
        self.is_alive = True
        for p in self.four_points:
            if self.pixel_at(int(p[0]), int(p[1])) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map=None):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.pixel_at(x, y) == (255, 255, 255, 255) and len < 200:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    """
    #------------------------------------------------------------------------------
    def draw_collision(self, screen):
        for i in range(4):
            x = int(self.four_points[i][0])
            y = int(self.four_points[i][1])
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 5)

    def check_radar_for_draw(self, degree, map=None):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 2000:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars_for_draw.append([(x, y), dist])
    """
    def check_checkpoint(self):
        p = check_point[self.current_check]
        self.prev_distance = self.cur_distance
        dist = get_distance(p, self.center)
        if dist < 70:
            self.current_check += 1
            self.prev_distance = 9999
            self.check_flag = True
            if self.current_check >= len(check_point):
                self.current_check = 0
                self.goal = True
            else:
                self.goal = False

        self.cur_distance = dist
    #------------------------------------------------------------------------------


    def update(self,map=None):
        #check speed
        self.speed -= 0.5
        if self.speed > 10: self.speed = 10
        if self.speed < 1:  self.speed = 1
        
        # required for NEAT
        if map is not None:
            self.speed = 7 # NEAT

        #check position
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        # required for NEAT
        if map is not None:
            self.check_collision(self.map)
            self.radars.clear()
            for d in range(-90, 120, 45):
                self.check_radar(d, self.map)

        """
        self.car.radars_for_draw.clear()
        for d in range(-90, 105, 15):
            self.car.check_radar_for_draw(d)
        pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        
        self.car.draw_collision(self.screen)
        # self.car.draw_radar(self.screen) # moved to car.draw()
        self.car.draw(self.screen)
        """

    #-------------------------------------------------------------------
    # required for NEAT
    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)
        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self):
        return self.distance / 50.0
    #-------------------------------------------------------------------

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


class PyRace2D:
    def __init__(self, is_render = True, car = True, mode = 0):
        # print('PyRace2D - INIT ENVIRONMENT')
        pygame.init()
        if is_render:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        else:
            self.screen = pygame.Surface((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.map = pygame.image.load('race_track_ie.png')
        self.cars = []
        if car:
            self.car = Car('car.png', self.map, [500, 650])
            self._car_surface = self.car.surface  # cache to avoid re-loading on reset
            self.cars.append(self.car)
        else:
            self._car_surface = None
        self.game_speed = 60*0 # as fast as possible...
        self.is_render = is_render
        if not isinstance(mode, int):
            mode = 0
        self.mode = mode  # 0: normal, 1:dark, 2: normal (force display)

    def reset_state(self, mode=0):
        """Reset car to start without reloading assets (no pygame.init, no image I/O)."""
        if not isinstance(mode, int):
            mode = 0
        self.mode = mode
        self.car = Car(None, self.map, [500, 650], surface=self._car_surface)
        self.cars = [self.car]

    def enable_display(self):
        """Switch from offscreen surface to a real pygame window (call once before evaluation)."""
        self.is_render = True
        self.screen = pygame.display.set_mode((screen_width, screen_height))

    def action(self, action):
        if action == 0: self.car.speed += 2
        elif action == 1: self.car.angle += 5
        elif action == 2: self.car.angle -= 5

        self.car.update()
        self.car.check_collision()
        self.car.check_checkpoint()

        self.car.radars.clear()
        for d in range(-90, 120, 45):
            self.car.check_radar(d)

    def evaluate(self):
        reward = 0
        """
        if self.car.check_flag:
            self.car.check_flag = False
            reward = 2000 - self.car.time_spent
            self.car.time_spent = 0
        """
        if not self.car.is_alive: # crash
            reward = -10000 + self.car.distance

        elif self.car.goal:
            # reward = 10000*(1+self.car.current_check)/len(check_point)
            reward = 10000
            # print('goal',self.car.current_check,len(check_point))
        return reward

    def is_done(self):
        if not self.car.is_alive or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    def observe(self):
        # return state
        radars = self.car.radars
        # print('RADARS',radars)
        ret = [0, 0, 0, 0, 0]
        i = 0
        for r in radars:
            ret[i] = int(r[1] / 20)
            i += 1

        return ret

    def view_(self, msgs=[]): # RENDERING...
        if not self.is_render:
            return
        if not isinstance(self.mode, int):
            self.mode = 0
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3
                if event.key == pygame.K_p:
                    self.mode += 1
                    self.mode = self.mode % 3
                elif event.key == pygame.K_q:
                    done = True
                    exit()

        self.screen.blit(self.map, (0, 0))

        if self.mode == 1:
            self.screen.fill((0, 0, 0))
        """
        self.car.radars_for_draw.clear()
        for d in range(-90, 105, 15):
            self.car.check_radar_for_draw(d)
        """
        if len(self.cars) == 1:
            pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        """
        self.car.draw_collision(self.screen)
        """
        # self.car.draw_radar(self.screen) # moved to car.draw()
        
        # self.car.draw(self.screen)
        for car in self.cars:
            if car.get_alive():
                car.draw(self.screen)

        # Display messages...
        for k,msg in enumerate(msgs):
            myfont = pygame.font.SysFont("impact", 20)
            label = myfont.render(msg, 1, (0,0,0))
            self.screen.blit(label,(1055,290+k*25))
            pass

        text = self.font.render("Press 'm' to change view mode", True, (255, 255, 0))
        text_rect = text.get_rect()
        # text_rect.center = (screen_width/2, 100)
        text_rect.topleft = (750,0)
        self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.game_speed)


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))


class PyRace2DV3(PyRace2D):
    """Improved environment used by Pyrace-v3.

    Changes vs PyRace2D:
      - BRAKE action (action=3): reduces speed by 2 (clamped to car minimum)
      - observe() returns raw float pixel distances (0–200) instead of bucketed ints 0–10
      - evaluate() adds a per-step speed reward and a checkpoint bonus for denser signal
    """

    def action(self, action):
        if action == 0:   self.car.speed += 2
        elif action == 1: self.car.angle += 5
        elif action == 2: self.car.angle -= 5
        elif action == 3: self.car.speed = max(1, self.car.speed - 2)  # brake
        self.car.update()
        self.car.check_collision()
        self.car.check_checkpoint()
        self.car.radars.clear()
        for d in range(-90, 120, 45):
            self.car.check_radar(d)

    def observe(self):
        # continuous pixel distances (0–200) — more informative than the 0–10 bucketed version
        ret = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i, r in enumerate(self.car.radars):
            ret[i] = float(r[1])
        return ret

    def evaluate(self):
        if not self.car.is_alive:
            return -10000 + self.car.distance
        if self.car.goal:
            return 10000
        # per-step speed reward makes the signal dense so the agent doesn't have to wait for crashes/goals
        reward = self.car.speed * 0.5
        if self.car.check_flag:
            self.car.check_flag = False
            reward += 500  # bonus for each checkpoint passed
        return reward
