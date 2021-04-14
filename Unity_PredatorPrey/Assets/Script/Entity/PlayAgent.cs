using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;

public class PlayAgent : Entity
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

    public override void Init()
    {
        base.Init();
        this.type = Entity_Type.AGENT;
    }

    public override void SetPostion(int x, int z)
    {
        trans.position = new Vector3(x, 1f, z);
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

        bool not_wall = true;
        bool not_other_agent = true;

        var hit = Physics.OverlapBox(
           targetPos, new Vector3(0.3f, 0.3f, 0.3f));

        if (hit.Where(col => col.gameObject.CompareTag("Wall")).ToArray().Length != 0)
            not_wall = false;

        if (hit.Where(col => col.gameObject.CompareTag("Agent")).ToArray().Length != 0)
            not_other_agent = false;

        if (not_wall && not_other_agent)
            trans.position = targetPos;
    }
}
