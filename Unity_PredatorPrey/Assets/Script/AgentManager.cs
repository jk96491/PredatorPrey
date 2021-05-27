using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AgentManager : Agent
{
    [SerializeField]
    private List<PlayAgent> agents = new List<PlayAgent>();

    EnvironmentParameters m_ResetParams;
    ObservationManager observationManager;
  
    public delegate void EpisodeBegin();
    public EpisodeBegin EpisodeBeginDel;

    public delegate void CalcReward(List<PlayAgent> agents);
    public CalcReward CalcRewardDel;


    public void InitDelgates(EpisodeBegin episodeBeginDel, CalcReward CalcRewardDel_)
    {
        EpisodeBeginDel = episodeBeginDel;
        CalcRewardDel = CalcRewardDel_;
    }

    public override void Initialize()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        observationManager = ObservationManager.Instance;
    }

    public void InitAgents(List<PlayAgent> agentList)
    {
        agents = agentList;
    }

    public void SetAgentPos(int index, int x, int z)
    {
        if (null == agents)
            return;

        if (agents.Count <= index)
            return;

        if (null != agents[index])
            agents[index].SetPostion(x, z);
    }

    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        for (int i = 0; i < agents.Count; i++)
        {
            var positionX = (int)agents[i].Trans.localPosition.x;
            var positionZ = (int)agents[i].Trans.localPosition.z;
            var maxPosition = 9;
            if (positionX <= -maxPosition)
            {
                actionMask.SetActionEnabled(i, PlayAgent.k_Left, false);
            }

            if (positionX >= maxPosition)
            {
                actionMask.SetActionEnabled(i, PlayAgent.k_Right, false);
            }

            if (positionZ <= -maxPosition)
            {
                actionMask.SetActionEnabled(i, PlayAgent.k_Down, false);
            }

            if (positionZ >= maxPosition)
            {
                actionMask.SetActionEnabled(i, PlayAgent.k_Up, false);
            }
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // state
        observationManager.CollectState();
        List<List<float>> states = observationManager.States;

        for(int i = 0; i < states.Count; i++)
        {
            for (int j = 0; j < states[i].Count; j++)
                sensor.AddObservation(states[i][j]);
        }
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        AddReward(-0.01f);

        var selectActions = actions.DiscreteActions;
        

        for (int i = 0; i < agents.Count; i++)
        {
            if(null != agents[i])
                agents[i].SetAction(selectActions[i]);
        }

        if (null != CalcRewardDel)
            CalcRewardDel(agents);
    }

    public override void OnEpisodeBegin()
    {
        for (int i = 0; i < agents.Count; i++)
            agents[i].SetActive(true);

        if (null != EpisodeBeginDel)
            EpisodeBeginDel();
    }
}
