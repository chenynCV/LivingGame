from agent import Agent
from planet import Planet
from rule import Rule
from render import plotAgentBelief


def main(args):
    survivors = []
    for g in range(args.generations):
        print('Start evolution {}.'.format(g))
        agents = [Agent() for _ in range(10)]
        rule = Rule(planet=Planet(Resources=1e3, N=10), agents=agents)
        for age in range(args.max_age):
            rule.tick(viz=args.viz)
            if age % args.print_freq == 0:
                print('generation {}, {} living!'.format(g, len(rule.agents)))
            if len(rule.agents) == 1:
                survivors.append(rule.agents[0])
                print("Age: {}, Entropy: {}!".format(
                    rule.agents[0].age, rule.agents[0].entropy))
                break
            elif len(rule.agents) == 0:
                break

    maxAge = 0
    oldestAgent = None
    for agent in survivors:
        if agent.age > maxAge:
            maxAge = agent.age
            oldestAgent = agent
    plotAgentBelief(oldestAgent)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Living Game')
    parser.add_argument('-g', '--generations', default=3,
                        type=int, metavar='N', help='number of agent generations')
    parser.add_argument('--print-freq', default=100,
                        type=int, help='print frequency')
    parser.add_argument('--viz', default=True,
                        help='Visualization game progress')
    parser.add_argument('--max-age', default=1000,
                        type=int, metavar='N', help='max age')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
