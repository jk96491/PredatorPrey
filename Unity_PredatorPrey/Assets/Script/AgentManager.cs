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

    public void InitDelgates(EpisodeBegin episodeBeginDel)
    {
        EpisodeBeginDel = episodeBeginDel;
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

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {

    }

    public override void OnEpisodeBegin()
    {
        for (int i = 0; i < agents.Count; i++)
            agents[i].SetActive(true);

        if (null != EpisodeBeginDel)
            EpisodeBeginDel();
    }
}
