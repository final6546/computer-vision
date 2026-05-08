import pygame
import math
import sys

# 初始化Pygame
pygame.init()

# 常量定义
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
FPS = 60

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
BLUE = (52, 152, 219)
RED = (231, 76, 60)
GREEN = (46, 204, 113)
ORANGE = (243, 156, 18)
PURPLE = (155, 89, 182)

COLORS = [BLUE, RED, GREEN, ORANGE, PURPLE]


class Shape:
    """形状类，支持矩形和图像"""
    def __init__(self, x, y, width, height, color, shape_type='rect', image=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.shape_type = shape_type
        self.image = image
        self.rotation = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.selected = False
        
    def get_corners(self):
        """获取旋转后的四个角点"""
        w = self.width * self.scale_x / 2
        h = self.height * self.scale_y / 2
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        
        rad = math.radians(self.rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_r - cy * sin_r + self.x
            ry = cx * sin_r + cy * cos_r + self.y
            rotated.append((rx, ry))
        
        return rotated
    
    def contains_point(self, px, py):
        """检查点是否在形状内"""
        dx = px - self.x
        dy = py - self.y
        
        rad = math.radians(-self.rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        
        local_x = dx * cos_r - dy * sin_r
        local_y = dx * sin_r + dy * cos_r
        
        w = self.width * self.scale_x / 2
        h = self.height * self.scale_y / 2
        
        return abs(local_x) <= w and abs(local_y) <= h
    
    def draw(self, surface):
        """绘制形状"""
        if self.shape_type == 'image' and self.image:
            # 绘制图像
            scaled_img = pygame.transform.scale(
                self.image, 
                (int(self.width * self.scale_x), int(self.height * self.scale_y))
            )
            rotated_img = pygame.transform.rotate(scaled_img, -self.rotation)
            rect = rotated_img.get_rect(center=(self.x, self.y))
            surface.blit(rotated_img, rect)
        else:
            # 绘制矩形
            corners = self.get_corners()
            pygame.draw.polygon(surface, self.color, corners)
        
        # 绘制选中边框
        if self.selected:
            corners = self.get_corners()
            pygame.draw.polygon(surface, RED, corners, 3)
            # 绘制中心点
            pygame.draw.circle(surface, RED, (int(self.x), int(self.y)), 5)


class Button:
    """按钮类"""
    def __init__(self, x, y, width, height, text, color=BLUE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover = False
        
    def draw(self, surface, font):
        color = tuple(min(c + 30, 255) for c in self.color) if self.hover else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Slider:
    """滑块类"""
    def __init__(self, x, y, width, min_val, max_val, initial_val, label):
        self.x = x
        self.y = y
        self.width = width
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.height = 20
        
    def draw(self, surface, font):
        # 绘制标签
        label_text = f"{self.label}: {self.value:.1f}"
        text_surf = font.render(label_text, True, BLACK)
        surface.blit(text_surf, (self.x, self.y - 20))
        
        # 绘制滑块轨道
        pygame.draw.rect(surface, GRAY, (self.x, self.y, self.width, self.height), border_radius=10)
        
        # 计算滑块位置
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        slider_x = self.x + ratio * self.width
        
        # 绘制滑块
        pygame.draw.circle(surface, BLUE, (int(slider_x), self.y + self.height // 2), 12)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            slider_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            if slider_rect.collidepoint(event.pos):
                self.dragging = True
                self.update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_value(event.pos[0])
            
    def update_value(self, mouse_x):
        ratio = max(0, min(1, (mouse_x - self.x) / self.width))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)


class TransformEditor:
    """2D变换编辑器主类"""
    
    def get_chinese_font(self, size):
        """获取支持中文的字体"""
        # 尝试常见的中文字体
        font_names = [
            'microsoftyaheimicrosoftyaheiui',  # 微软雅黑
            'microsoftyahei',
            'simsun',  # 宋体
            'simhei',  # 黑体
            'msgothic',  # MS Gothic (日文，但支持中文)
            'arial',
        ]
        
        # 在Windows上尝试系统字体
        for font_name in font_names:
            try:
                font = pygame.font.SysFont(font_name, size)
                # 测试是否支持中文
                test_surface = font.render('测试', True, (0, 0, 0))
                if test_surface.get_width() > 0:
                    return font
            except:
                continue
        
        # 如果都失败，返回默认字体
        return pygame.font.Font(None, size)
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2D Transform Editor")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # 字体 - 使用系统中文字体
        self.font = self.get_chinese_font(24)
        self.title_font = self.get_chinese_font(32)
        
        # 形状列表
        self.shapes = []
        self.selected_shape = None
        
        # 拖动状态
        self.dragging = False
        self.rotating = False
        self.drag_start_pos = (0, 0)
        self.drag_start_rotation = 0
        
        # 创建UI元素
        self.create_ui()
        
        # 颜色索引
        self.color_index = 0
        
    def create_ui(self):
        """创建UI元素"""
        panel_x = 20
        
        # 按钮
        self.buttons = [
            Button(panel_x, 60, 200, 40, "Add Rectangle", BLUE),
            Button(panel_x, 110, 200, 40, "Add Image", GREEN),
            Button(panel_x, 160, 200, 40, "Delete", RED),
            Button(panel_x, 210, 200, 40, "Clear All", ORANGE),
        ]
        
        # 滑块
        slider_y = 280
        slider_spacing = 60
        self.sliders = [
            Slider(panel_x, slider_y, 200, -400, 400, 0, "Position X"),
            Slider(panel_x, slider_y + slider_spacing, 200, -400, 400, 0, "Position Y"),
            Slider(panel_x, slider_y + slider_spacing * 2, 200, 0, 360, 0, "Rotation"),
            Slider(panel_x, slider_y + slider_spacing * 3, 200, 0.1, 3, 1, "Scale X"),
            Slider(panel_x, slider_y + slider_spacing * 4, 200, 0.1, 3, 1, "Scale Y"),
        ]
        
    def add_shape(self, shape_type='rect'):
        """添加形状"""
        x = WINDOW_WIDTH // 2
        y = WINDOW_HEIGHT // 2
        color = COLORS[self.color_index % len(COLORS)]
        self.color_index += 1
        
        if shape_type == 'image':
            # 创建一个简单的渐变图像
            image = pygame.Surface((100, 100))
            for i in range(100):
                color_val = int(255 * i / 100)
                pygame.draw.rect(image, (color_val, 100, 255 - color_val), (0, i, 100, 1))
            shape = Shape(x, y, 100, 100, color, 'image', image)
        else:
            shape = Shape(x, y, 100, 100, color, 'rect')
        
        self.shapes.append(shape)
        
    def delete_selected(self):
        """删除选中的形状"""
        if self.selected_shape:
            self.shapes.remove(self.selected_shape)
            self.selected_shape = None
            
    def clear_canvas(self):
        """清空画布"""
        self.shapes.clear()
        self.selected_shape = None
        
    def update_sliders_from_shape(self):
        """从选中的形状更新滑块值"""
        if self.selected_shape:
            canvas_center_x = WINDOW_WIDTH // 2
            canvas_center_y = WINDOW_HEIGHT // 2
            
            self.sliders[0].value = self.selected_shape.x - canvas_center_x
            self.sliders[1].value = self.selected_shape.y - canvas_center_y
            self.sliders[2].value = self.selected_shape.rotation
            self.sliders[3].value = self.selected_shape.scale_x
            self.sliders[4].value = self.selected_shape.scale_y
            
    def update_shape_from_sliders(self):
        """从滑块更新选中的形状"""
        if self.selected_shape:
            canvas_center_x = WINDOW_WIDTH // 2
            canvas_center_y = WINDOW_HEIGHT // 2
            
            self.selected_shape.x = self.sliders[0].value + canvas_center_x
            self.selected_shape.y = self.sliders[1].value + canvas_center_y
            self.selected_shape.rotation = self.sliders[2].value
            self.selected_shape.scale_x = self.sliders[3].value
            self.selected_shape.scale_y = self.sliders[4].value

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            # 处理按钮事件
            for i, button in enumerate(self.buttons):
                if button.handle_event(event):
                    if i == 0:  # 添加矩形
                        self.add_shape('rect')
                    elif i == 1:  # 添加图像
                        self.add_shape('image')
                    elif i == 2:  # 删除选中
                        self.delete_selected()
                    elif i == 3:  # 清空画布
                        self.clear_canvas()
                        
            # 处理滑块事件
            for slider in self.sliders:
                slider.handle_event(event)
                
            # 处理画布上的鼠标事件
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[0] > 250:  # 只在画布区域
                    # 检查是否点击了形状
                    clicked_shape = None
                    for shape in reversed(self.shapes):
                        if shape.contains_point(*event.pos):
                            clicked_shape = shape
                            break
                    
                    # 更新选中状态
                    for shape in self.shapes:
                        shape.selected = False
                    
                    if clicked_shape:
                        clicked_shape.selected = True
                        self.selected_shape = clicked_shape
                        self.dragging = True
                        self.rotating = pygame.key.get_mods() & pygame.KMOD_SHIFT
                        self.drag_start_pos = event.pos
                        self.drag_start_rotation = clicked_shape.rotation
                        self.update_sliders_from_shape()
                    else:
                        self.selected_shape = None
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging = False
                self.rotating = False
                
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging and self.selected_shape:
                    if self.rotating:
                        # 旋转模式
                        dx = event.pos[0] - self.selected_shape.x
                        dy = event.pos[1] - self.selected_shape.y
                        angle = math.degrees(math.atan2(dy, dx))
                        
                        start_dx = self.drag_start_pos[0] - self.selected_shape.x
                        start_dy = self.drag_start_pos[1] - self.selected_shape.y
                        start_angle = math.degrees(math.atan2(start_dy, start_dx))
                        
                        self.selected_shape.rotation = self.drag_start_rotation + (angle - start_angle)
                    else:
                        # 移动模式
                        dx = event.pos[0] - self.drag_start_pos[0]
                        dy = event.pos[1] - self.drag_start_pos[1]
                        self.selected_shape.x += dx
                        self.selected_shape.y += dy
                        self.drag_start_pos = event.pos
                    
                    self.update_sliders_from_shape()
                    
        # 从滑块更新形状
        self.update_shape_from_sliders()
        
    def draw_grid(self):
        """绘制网格"""
        grid_color = (230, 230, 230)
        for x in range(250, WINDOW_WIDTH, 50):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, 50):
            pygame.draw.line(self.screen, grid_color, (250, y), (WINDOW_WIDTH, y))
            
    def draw(self):
        """绘制所有元素"""
        self.screen.fill(WHITE)
        
        # 绘制网格
        self.draw_grid()
        
        # 绘制侧边栏背景
        pygame.draw.rect(self.screen, LIGHT_GRAY, (0, 0, 240, WINDOW_HEIGHT))
        
        # 绘制标题
        title = self.title_font.render("2D Transform Editor", True, BLACK)
        self.screen.blit(title, (20, 20))
        
        # 绘制按钮
        for button in self.buttons:
            button.draw(self.screen, self.font)
            
        # 绘制滑块
        for slider in self.sliders:
            slider.draw(self.screen, self.font)
            
        # 绘制形状
        for shape in self.shapes:
            shape.draw(self.screen)
            
        # 绘制提示信息
        hints = [
            "Tips:",
            "* Click to select",
            "* Drag to move",
            "* Shift+Drag to rotate",
            "* Use sliders for precision",
        ]
        hint_y = WINDOW_HEIGHT - 120
        for hint in hints:
            hint_surf = self.font.render(hint, True, BLACK)
            self.screen.blit(hint_surf, (20, hint_y))
            hint_y += 25
            
        pygame.display.flip()
        
    def run(self):
        """主循环"""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    editor = TransformEditor()
    editor.run()
