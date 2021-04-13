using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObservationManager : MonoSingleton<ObservationManager>
{
    [SerializeField]
    List<Goal> goals = null;
    [SerializeField]
    private List<PlayAgent> agents = new List<PlayAgent>();

    private List<List<float>> stateInfos = null;
    private List<List<float>> obsInfos = null;

    public List<Goal> Goals { get { return goals; } }
    public List<List<float>> States { get { return stateInfos; } }
    public List<List<float>> Obs { get { return obsInfos; } }

    private float map_size;

    public void init(int map_size_)
    {
        map_size = map_size_;
    }

    public void InitAgents(List<PlayAgent> agentList)
    {
        agents = agentList;
    }

    public void InitGoals(List<Goal> goalsList)
    {
        goals = goalsList;
    }

    public void CollectState()
    {
        int length = agents.Count + goals.Count;

        if (null == stateInfos)
            stateInfos = new List<List<float>>();
        else
            stateInfos.Clear();

        for (int i = 0; i < length; i++)
        {
            List<float> curInfo = null;

            if (i < agents.Count)
                curInfo = GetObjectInfo(agents[i]);
            else
            {
                int index = i - agents.Count;
                curInfo = GetObjectInfo(goals[index]);
            }

            stateInfos.Add(curInfo);
        }
    }

    public void CollectObservation()
    {
        if (null == obsInfos)
            obsInfos = new List<List<float>>();
        else
            obsInfos.Clear();

        for (int index = 0; index < agents.Count; index++)
        {
            int agent_count = 0;
            int goal_count = 0;

            List<float> OwnInfo = GetObjectInfo(agents[index]);
            List<float> curAgentInfo = null;
            List<float> curGoalInfo = null;
            //obsInfos.Add(OwnInfo);

            // Agent ����
            for (int i = 0; i < agents.Count; i++)
            {
                if (i == index)
                    continue;

                float srcAgentX = agents[index].Trans.position.x;
                float srcAgentZ = agents[index].Trans.position.z;

                float destAgentX = agents[i].Trans.position.x;
                float destAgentZ = agents[i].Trans.position.z;

                float dis = Mathf.Sqrt((destAgentX - srcAgentX) * (destAgentX - srcAgentX) + (destAgentZ - srcAgentZ) * (destAgentZ - srcAgentZ));

                if(dis <= 5)
                {
                    curAgentInfo = GetObjectInfo(agents[i]);
                    agent_count++;
                    OwnInfo.AddRange(curAgentInfo);
                }
            }

            // Goal ����
            for (int i = 0; i < goals.Count; i++)
            {
                if (i == index)
                    continue;

                float AgentX = agents[index].Trans.position.x;
                float AgentZ = agents[index].Trans.position.z;

                float GoalX = goals[i].Trans.position.x;
                float GoalZ = goals[i].Trans.position.z;

                float dis = Mathf.Sqrt((GoalX - AgentX) * (GoalX - AgentX) + (GoalZ - AgentZ) * (GoalZ - AgentZ));

                if (dis <= 5)
                {
                    curGoalInfo = GetObjectInfo(goals[i]);
                    goal_count++;
                    OwnInfo.AddRange(curGoalInfo);
                }
            }

            int agent_remain = (agents.Count - 1) - agent_count;
            int goal_remain = goals.Count - goal_count;   

            for (int i = 0; i < agent_remain + goal_remain; i++)
            {
                List<float> emptyInfo = new List<float>();

                emptyInfo.Add(0);
                emptyInfo.Add(0);
                emptyInfo.Add(0);
                emptyInfo.Add(0);
                OwnInfo.AddRange(emptyInfo);
            }

            obsInfos.Add(OwnInfo);
        }
    }

    public List<float> GetObjectInfo(Entity obj)
    {
        List<float> curInfo = new List<float>();

        if(!obj.IsActive)
        {
            curInfo.Add(0);
            curInfo.Add(0);
            curInfo.Add(0);
            curInfo.Add(0);

            return curInfo;
        }

        if (obj.Type == Entity.Entity_Type.AGENT)
        {
            curInfo.Add(1);
            curInfo.Add(0);
        }
        else
        {
            curInfo.Add(0);
            curInfo.Add(1);
        }

        float posX = obj.Trans.position.x;
        float posZ = obj.Trans.position.z;

        float dis = map_size;

        curInfo.Add(posX / dis);
        curInfo.Add(posZ / dis);

        return curInfo;
    }
}
