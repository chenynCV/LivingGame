from agent import Agent
from planet import Planet
from rule import Rule


def main(args):
    agents = [Agent() for _ in range(10)]
    rule = Rule(planet=Planet(split='val'), agents=agents)
    for age in range(0, args.max_age):
        if age % args.print_freq == 0:
            rule.tick(args.viz)
            print('age {}, acc {}, {} living!'.format(
                age, rule.agents[0].acc, len(rule.agents)))
        else:
            rule.tick()
        if len(rule.agents) == 0:
            break


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Living Game')
    parser.add_argument('--print-freq', default=100,
                        type=int, help='print frequency')
    parser.add_argument('--viz', default=False,
                        help='Visualization game progress')
    parser.add_argument('--max-age', default=100000000,
                        type=int, metavar='N', help='max age')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
