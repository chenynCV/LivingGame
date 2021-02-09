from agent import Agent
from planet import Planet
from rule import Rule
from render import plotAgentBelief


def main():
    survivors = []

    for _ in range(3):
        agents = [Agent() for _ in range(10)]
        rule = Rule(planet=Planet(Resources=1e3, N=10), agents=agents)
        for cnt in range(1000):
            rule.tick(viz=True)
            if len(rule.agents) <= 1:
                if len(rule.agents) > 0:
                    survivors.append(rule.agents[0])
                print("Age: {}, Entropy: {}!".format(
                    rule.agents[0].age, rule.agents[0].entropy))
                break

    maxAge = 0
    oldestAgent = None
    for agent in survivors:
        if agent.age > maxAge:
            maxAge = agent.age
            oldestAgent = agent
    plotAgentBelief(oldestAgent)


if __name__ == '__main__':
    main()
