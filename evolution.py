from tqdm import tqdm
from agent import Agent
from planet import Planet
from rule import Rule
from render import plotAgentBelief


def main(args):
    survivors = []
    for g in tqdm(range(args.generations)):
        agents = [Agent() for _ in range(3)]
        rule = Rule(planet=Planet(Resources=300, N=10),
                    agents=agents, viz=args.viz)
        for age in range(1, args.max_age):
            rule.tick()
            if age % args.print_freq == 0:
                print('generation {}, age {}, {} living!'.format(
                    g, age, len(rule.agents)))
            if len(rule.agents) == 0:
                break
        if rule._agent is not None:
            survivors.append(rule._agent)

    oldestAgent = None
    maxAge = 0
    for agent in survivors:
        if agent.age > maxAge:
            maxAge = agent.age
            oldestAgent = agent
    print('The oldest Agent, age {}!'.format(oldestAgent.age))
    plotAgentBelief(oldestAgent)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Living Game')
    parser.add_argument('-g', '--generations', default=300000,
                        type=int, metavar='N', help='number of agent generations')
    parser.add_argument('--print-freq', default=1000,
                        type=int, help='print frequency')
    parser.add_argument('--viz', default=False,
                        help='Visualization game progress')
    parser.add_argument('--max-age', default=1000,
                        type=int, metavar='N', help='max age')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
