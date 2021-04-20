using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using Unity.MLAgents.Sensors;
using Unity.MLAgents;

public class PlayAgent : Agent
{
    public const int k_NoAction = 0;  // do nothing!
    public const int k_Up = 1;
    public const int k_Down = 2;
    public const int k_Left = 3;
    public const int k_Right = 4;

    private Vector3 moveRight = new Vector3(1f, 0, 0);
    private Vector3 moveLeft = new Vector3(-1f, 0, 0);
    private Vector3 moveUp = new Vector3(0, 0, 1f);
    private Vector3 moveDown = new Vector3(0, 0, -1f);

    protected Entity.Entity_Type type;
    public Entity.Entity_Type Type { get { return this.type; } }

    protected Transform trans = null;
    public Transform Trans { get { return this.trans; } }

    protected int index = 0;
    public int Index { get { return index; } }

    protected bool isActive = false;
    public bool IsActive { get { return this.isActive; } }

    public virtual void Init(int index_ = -1)
    {
        this.type = Entity.Entity_Type.AGENT;
        trans = gameObject.transform;
        index = index_;
    }

    EnvironmentParameters m_ResetParams;
    protected ObservationManager observationManager;

    public override void Initialize()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        observationManager = ObservationManager.Instance;
    }

    public virtual void SetPostion(int x, int z)
    {
        trans.position = new Vector3(x, 1f, z);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        List<float> obs = observationManager.CollectObservation(Index);

        for (int i = 0; i < obs.Count; i++)
            sensor.AddObservation(obs[i]);
    }

    public void SetAction(int action)
    {
        var targetPos = trans.position;
        switch (action)
        {
            case k_NoAction:
                // do nothing
                break;
            case k_Right:
                targetPos = trans.position + moveRight;
                break;
            case k_Left:
                targetPos = trans.position + moveLeft;
                break;
            case k_Up:
                targetPos = trans.position + moveUp;
                break;
            case k_Down:
                targetPos = trans.position + moveDown;
                break;
            default:
                throw new ArgumentException("Invalid action value");
        }
        
        bool not_other_agent = true;

        var hit = Physics.OverlapBox(
           targetPos, new Vector3(0.3f, 0.3f, 0.3f));

        if (hit.Where(col => col.gameObject.CompareTag("Agent")).ToArray().Length != 0)
            not_other_agent = false;

        if (not_other_agent)
            trans.position = targetPos;
    }

    public void SetActive(bool active)
    {
        isActive = active;
        gameObject.SetActive(active);
    }
}
