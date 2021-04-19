using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;

public class Area : MonoBehaviour
{
    [SerializeField]
    private int maxAgentNums = 0;
    [SerializeField]
    private int maxGoalNums = 0;
    [SerializeField]
    private AgentManager agentManager = null;
    [SerializeField]
    private Transform goalParentsTrans = null;
    [SerializeField]
    private Transform AgentMangerTrans = null;
    [SerializeField]
    private List<Goal> goals = new List<Goal>();

    private List<Transform> goalTrans = new List<Transform>();

    EnvironmentParameters m_ResetParams;
    private int objectCount = 0;
    private int mapSize = 9;

    void Start()
    {
        objectCount = maxAgentNums + maxGoalNums;
        m_ResetParams = Academy.Instance.EnvironmentParameters;

        ObservationManager.Instance.init(mapSize * 2);

        CreateAgents();
        CreatGoals();

        agentManager.InitDelgates(EpisodeBegin, CalcReward);
    }

    private void CreateAgents()
    {
        List<PlayAgent> agentList = new List<PlayAgent>();

        for (int i = 0; i < maxAgentNums; i++)
        {
            GameObject agentprefab = Resources.Load<GameObject>("prefabs/Agent");

            if (null == agentprefab)
                continue;

            GameObject agent = Instantiate<GameObject>(agentprefab);

            if (null == agent)
                continue;

            agent.name = string.Format("Agent{0}", i + 1);

            if (null != AgentMangerTrans)
                agent.transform.SetParent(AgentMangerTrans);

            PlayAgent agentIns = agent.GetComponent<PlayAgent>();

            if (null != agentIns)
            {
                agentIns.Initialize();
                agentIns.Init(i);
                agentList.Add(agentIns);
            }
        }

        if (null != agentManager)
            agentManager.InitAgents(agentList);

        ObservationManager.Instance.InitAgents(agentList);
    }

    private void CreatGoals()
    {
        List<Goal> goalList = new List<Goal>();
        for (int i = 0; i < maxGoalNums; i++)
        {
            GameObject goalprefab = Resources.Load<GameObject>("prefabs/Goal");

            if (null == goalprefab)
                continue;

            GameObject goal = Instantiate<GameObject>(goalprefab);

            if (null == goal)
                continue;

            goal.name = string.Format("Goal{0}", i + 1);

            if (null != goalParentsTrans)
                goal.transform.SetParent(goalParentsTrans);

            Goal goalIns = goal.GetComponent<Goal>();

            if (null != goalIns)
            {
                goalIns.Init();
                goalList.Add(goalIns);
                goalTrans.Add(goalIns.Trans);
            }
        }

        ObservationManager.Instance.InitGoals(goalList);
        goals = goalList;
    }

    private void ResetEnv()
    {
        ResetPostion();

        for (int i = 0; i < goals.Count; i++)
            goals[i].SetActive(true);

        step_count = 0;
    }

    public void ResetPostion()
    {
        var numbers = new HashSet<int>();
        while (numbers.Count < objectCount + 1)
            numbers.Add(Random.Range(-mapSize * mapSize, mapSize * mapSize));

        var numbersA = numbers.ToArray();

        for (int i = 0; i < objectCount; i++)
        {
            var x = (numbersA[i]) / mapSize;
            var z = (numbersA[i]) % mapSize;

            if (i < maxAgentNums)
            {
                agentManager.SetAgentPos(i, x, z);
            }
            else
            {
                int index = i - maxAgentNums;
                goalTrans[index].position = new Vector3(x, 0.5f, z);
            }
        }
    }

    int step_count = 0;
    private void EpisodeBegin()
    {
        ResetEnv();
    }

    private void CalcReward(List<PlayAgent> agents)
    {
        step_count++;
        for (int agent_index = 0; agent_index < agents.Count; agent_index++)
        {
            if (null == agents[agent_index])
                continue;

            for (int goal_index = 0; goal_index <  goals.Count; goal_index++)
            {
                if (null == goals[goal_index])
                    continue;

                Transform agentTrans = agents[agent_index].Trans;
                Transform goalTrans = goals[goal_index].Trans;

                float dis = Utils.get_distance(agentTrans, goalTrans);

                if(dis < 0.5f && goals[goal_index].IsActive)
                {
                    goals[goal_index].SetActive(false);
                    agentManager.AddReward(2f);
                }
            }
        }

        int obtain_count = 0;
        for (int goal_index = 0; goal_index < goals.Count; goal_index++)
        {
            if (null == goals[goal_index])
                continue;

            if (!goals[goal_index].IsActive)
                obtain_count++;
        }

        if(obtain_count == maxGoalNums)
        {
            agentManager.AddReward(10f);
            agentManager.AddReward(((160f - step_count) / 160f) * 5f);
            agentManager.EndEpisode();
        }

    }
}
