from agent import Agent
from planet import Planet
from rule import Rule


def main(args):
    agents = [Agent() for _ in range(10)]
    rule = Rule(planet=Planet(Resources=3000, N=10),
                agents=agents, viz=args.viz)
    for age in range(1, args.max_age):
        rule.tick()
        if age % args.print_freq == 0:
            print('age {}, {} living!'.format(age, len(rule.agents)))
        if len(rule.agents) == 0:
            break


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Living Game')
    parser.add_argument('--print-freq', default=10,
                        type=int, help='print frequency')
    parser.add_argument('--viz', default=True,
                        help='Visualization game progress')
    parser.add_argument('--max-age', default=10000,
                        type=int, metavar='N', help='max age')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
