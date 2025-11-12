import pygame
import sys

# 初始化
pygame.init()


# 屏幕
WIDTH, HEIGHT = 400, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("按钮计数器")

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 150, 255)
DARK_BLUE = (70, 120, 200)

# 字体
font = pygame.font.SysFont("notosanscjksc", 22)

# 按钮属性
button_rect = pygame.Rect(150, 120, 100, 50)

# 初始计数
count = 0

# 主循环
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                count += 1  # 点击按钮时计数加一

    # 绘制
    screen.fill(WHITE)

    # 按钮颜色变化（悬停效果）
    mouse_pos = pygame.mouse.get_pos()
    if button_rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, DARK_BLUE, button_rect)
    else:
        pygame.draw.rect(screen, BLUE, button_rect)

    # 按钮文字
    btn_text = font.render("点击", True, WHITE)
    screen.blit(btn_text, (button_rect.centerx - btn_text.get_width() // 2,
                           button_rect.centery - btn_text.get_height() // 2))

    # 显示计数
    count_text = font.render(str(count), True, BLACK)
    screen.blit(count_text, (WIDTH // 2 - count_text.get_width() // 2, 50))
    print(count)

    pygame.display.flip()
