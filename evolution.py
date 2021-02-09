from agent import Agent
from planet import Planet
from rule import Rule


def main():
    rule = Rule(planet=Planet(Resources=1e6, N=100))
    rule.agents = [Agent() for _ in range(10)]

    for cnt in range(100000):
        rule.tick()
        if cnt % 10000 == 0:
            print("{}: {} living!".format(cnt, len(rule.agents)))


if __name__ == '__main__':
    main()
