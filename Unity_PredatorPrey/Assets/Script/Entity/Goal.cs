using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Goal : Entity
{
    public override void Init(int index_ = -1)
    {
        base.Init(index_);
        this.type = Entity_Type.GOAL;
    }

    public override void SetPostion(int x, int z)
    {
        trans.position = new Vector3(x, 0.5f, z);
    }
}
