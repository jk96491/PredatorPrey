using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Entity : MonoBehaviour
{
    public enum Entity_Type : int
    {
        NONE =-1,
        AGENT = 0,
        GOAL = 1
    }

    protected Transform trans = null;
    public Transform Trans { get { return this.trans; } }

    protected Entity_Type type;
    public Entity_Type Type { get { return this.type; } }

    protected bool isActive = false;
    public bool IsActive { get { return this.isActive; } }

    public virtual void Init()
    {
        type = Entity_Type.NONE;
        trans = gameObject.transform;

        isActive = false;
    }

    public virtual void SetPostion(int x, int z)
    {
        trans.position = new Vector3(x, 0.5f, z);
    }

    public void SetActive(bool active)
    {
        isActive = active;
    }
}
