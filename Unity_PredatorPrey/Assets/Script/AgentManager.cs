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

    float m_TimeSinceDecision;
    public float timeBetweenDecisionsAtInference;

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
        // Mask the necessary actions if selected by the user.
        for (int i = 0; i < agents.Count; i++)
        {
            var positionX = (int)agents[i].Trans.localPosition.x;
            var positionZ = (int)agents[i].Trans.localPosition.z;
            var maxPosition = 9;
            if (positionX == -maxPosition)
            {
                actionMask.WriteMask(i, new[] { PlayAgent.k_Left });
            }

            if (positionX == maxPosition)
            {
                actionMask.WriteMask(i, new[] { PlayAgent.k_Right });
            }

            if (positionZ == -maxPosition)
            {
                actionMask.WriteMask(i, new[] { PlayAgent.k_Down });
            }

            if (positionZ == maxPosition)
            {
                actionMask.WriteMask(i, new[] { PlayAgent.k_Up });
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

        // obs
        observationManager.CollectObservation();
        List<List<float>> Obs = observationManager.Obs;

        for (int i = 0; i < Obs.Count; i++)
        {
            for (int j = 0; j < Obs[i].Count; j++)
                sensor.AddObservation(Obs[i][j]);
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

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    void WaitTimeInference()
    {
        if (Academy.Instance.IsCommunicatorOn)
        {
            RequestDecision();
        }
        else
        {
            if (m_TimeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                m_TimeSinceDecision = 0f;
                RequestDecision();
            }
            else
            {
                m_TimeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }
}
