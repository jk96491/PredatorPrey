using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayAgent : Entity
{
    public override void Init()
    {
        base.Init();
        this.type = Entity_Type.AGENT;
    }

    public override void SetPostion(int x, int z)
    {
        trans.position = new Vector3(x, 1f, z);
    }
}
