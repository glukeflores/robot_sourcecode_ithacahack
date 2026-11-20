# Hello world: a full tennis game in the terminal that ends with a giant rainbow "HELLO WORLD"
import math
import random
import shutil
import sys
import time

COLS = shutil.get_terminal_size().columns
WIDTH = max(40, min(COLS, 72))
HEIGHT = 12
GROUND = HEIGHT - 1
NET = WIDTH // 2
PLAYER_X = (2, WIDTH - 3)
PLAYERS = ("LEFTY", "RIGHTY")
PLAYER_COLORS = ("96", "95")
DELAY = 0.025

TRAIL_SHADES = ("38;5;238", "38;5;241", "38;5;244", "38;5;248", "38;5;252")
RAINBOW = (196, 202, 208, 214, 220, 226, 190, 154, 118, 82, 46, 47, 48, 49, 50,
           51, 45, 39, 33, 27, 21, 57, 93, 129, 165, 201, 200, 199, 198, 197)

WINNER_CALLS = ("WHAT A SHOT! 🔥", "Unreturnable! ⚡", "Clean winner! 💥", "Down the line! 🎯")
NET_CALLS = ("Into the net... 😬", "Caught the tape! 🫣", "Ooooh, so close! 😩")

FONT = {
    "H": ["█   █", "█   █", "█████", "█   █", "█   █"],
    "E": ["█████", "█    ", "████ ", "█    ", "█████"],
    "L": ["█    ", "█    ", "█    ", "█    ", "█████"],
    "O": [" ███ ", "█   █", "█   █", "█   █", " ███ "],
    "W": ["█   █", "█   █", "█ █ █", "██ ██", "█   █"],
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "D": ["████ ", "█   █", "█   █", "█   █", "████ "],
    " ": ["   "] * 5,
}


def c(text, code):
    return f"\033[{code}m{text}\033[0m"


def show(lines):
    sys.stdout.write("\033[H" + "\n".join(line + "\033[K" for line in lines) + "\n\033[J")
    sys.stdout.flush()


def score_text(points):
    a, b = points
    if a >= 3 and b >= 3:
        if a == b:
            return "DEUCE"
        return f"ADVANTAGE {PLAYERS[0 if a > b else 1]}"
    names = ("0", "15", "30", "40")
    return f"{names[a]} - {names[b]}"


def render(ball, trail, swinging, header, message):
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]
    for x in range(WIDTH):
        grid[GROUND][x] = c("═", "32")
    for y in range(GROUND - 4, GROUND):
        grid[y][NET] = c("┃", "97")

    for i, px in enumerate(PLAYER_X):
        grid[GROUND - 2][px] = c("O", PLAYER_COLORS[i] + ";1")
        grid[GROUND - 1][px] = c("█", PLAYER_COLORS[i])
        racket_x = px + 1 if i == 0 else px - 1
        if swinging[i]:
            grid[GROUND - 3][racket_x] = c("/" if i == 0 else "\\", "97;1")
        else:
            grid[GROUND - 1][racket_x] = c("|", "37")

    for shade, (tx, ty) in zip(TRAIL_SHADES[-len(trail):], trail):
        if 0 <= tx < WIDTH and 0 <= ty < GROUND and grid[ty][tx] == " ":
            grid[ty][tx] = c("·", shade)

    if ball:
        bx, by = ball
        if 0 <= bx < WIDTH:
            grid[GROUND][bx] = c("═", "90")  # shadow
            if 0 <= by < GROUND:
                grid[by][bx] = c("●", "93;1")

    return [header, message, ""] + ["".join(row) for row in grid]


def flight(x0, x1, peak, steps, y0=GROUND - 3, y1=GROUND - 3):
    path = []
    for i in range(steps + 1):
        t = i / steps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t - math.sin(t * math.pi) * peak
        path.append((round(x), round(y)))
    return path


def animate(path, hitter, header, message, trail):
    for i, pos in enumerate(path):
        swinging = [False, False]
        if i < 4:
            swinging[hitter] = True
        show(render(pos, trail, swinging, header, message))
        trail.append(pos)
        del trail[:-len(TRAIL_SHADES)]
        time.sleep(DELAY)


def play_point(points, server):
    header = (f" 🎾 {c(PLAYERS[0], PLAYER_COLORS[0] + ';1')}   "
              f"{c(score_text(points), '97;1')}   {c(PLAYERS[1], PLAYER_COLORS[1] + ';1')}")
    hitter, trail = server, []
    rally_length = random.randint(2, 6)

    for shot in range(rally_length):
        x0 = PLAYER_X[hitter] + (1 if hitter == 0 else -1)
        x1 = PLAYER_X[1 - hitter] + (-1 if hitter == 0 else 1)
        peak = random.uniform(4, HEIGHT - 5)
        steps = random.randint(18, 30)
        message = c(f"   Rally: {shot + 1}", "90")
        animate(flight(x0, x1, peak, steps), hitter, header, message, trail)
        hitter = 1 - hitter

    x0 = PLAYER_X[hitter] + (1 if hitter == 0 else -1)
    if random.random() < 0.65:
        # Winner: a flat, fast shot that blows past the opponent
        x1 = WIDTH + 4 if hitter == 0 else -5
        message = c("   " + random.choice(WINNER_CALLS), "93;1")
        animate(flight(x0, x1, 3, 14), hitter, header, message, trail)
        point_winner = hitter
    else:
        # Error: a low shot that dies in the net
        x1 = NET - 1 if hitter == 0 else NET + 1
        message = c("   " + random.choice(NET_CALLS), "91;1")
        animate(flight(x0, x1, 1.5, 12), hitter, header, message, trail)
        for y in range(GROUND - 3, GROUND):
            show(render((x1, y), [], [False, False], header, message))
            time.sleep(DELAY * 3)
        point_winner = 1 - hitter

    time.sleep(0.6)
    return point_winner


def play_game():
    points = [0, 0]
    server = 0
    while True:
        points[play_point(points, server)] += 1
        a, b = points
        if max(a, b) >= 4 and abs(a - b) >= 2:
            return 0 if a > b else 1


def banner_rows(words):
    rows = []
    for word in words:
        for r in range(5):
            rows.append(" ".join(FONT[ch][r] for ch in word))
        rows.append("")
    return rows


def finale(winner):
    text = "HELLO WORLD"
    rows = banner_rows([text]) if len(text) * 6 + 2 < COLS else banner_rows(["HELLO", "WORLD"])
    trophy = c(f"  🏆 {PLAYERS[winner]} wins the game! 🏆", PLAYER_COLORS[winner] + ";1")
    for frame in range(60):
        lines = [trophy, ""]
        for r, row in enumerate(rows):
            lines.append("  " + "".join(
                c(ch, f"38;5;{RAINBOW[(x + r * 2 - frame) % len(RAINBOW)]};1") if ch != " " else " "
                for x, ch in enumerate(row)
            ))
        show(lines)
        time.sleep(0.04)


def main():
    if not sys.stdout.isatty():
        print("hello world")
        return
    sys.stdout.write("\033[2J\033[?25l")  # clear screen, hide cursor
    try:
        winner = play_game()
        finale(winner)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # show cursor again


if __name__ == "__main__":
    main()
