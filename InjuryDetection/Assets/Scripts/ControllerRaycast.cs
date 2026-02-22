using System;
using UnityEngine;

public class ControllerRaycast : MonoBehaviour
{
    /// <summary>Fired when a subject bounding box is selected (or null on deselect).</summary>
    public static event Action<SubjectBox> OnSubjectSelected;

    [Header("References")]
    [SerializeField] private BraceWebSocket braceWs;

    [Header("Beam")]
    [SerializeField] private float maxRayLength = 20f;
    [SerializeField] private float beamWidth = 0.005f;
    [SerializeField] private Color beamColor = new Color(0.5f, 1f, 1f, 0.8f);
    [SerializeField] private Color beamHitColor = new Color(1f, 1f, 1f, 1f);

    [Header("Hover")]
    [SerializeField] private float hoverLineWidth = 0.01f;
    [SerializeField] private float normalLineWidth = 0.005f;

    private LineRenderer _line;
    private SubjectBox _hoveredBox;
    private SubjectBox _selectedBox;

    private void Start()
    {
        // Auto-find BraceWebSocket
        if (braceWs == null)
            braceWs = FindAnyObjectByType<BraceWebSocket>();

        _line = GetComponent<LineRenderer>();
        if (_line == null)
            _line = gameObject.AddComponent<LineRenderer>();

        _line.positionCount = 2;
        _line.startWidth = beamWidth;
        _line.endWidth = beamWidth * 0.5f;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Sprites/Default");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");

        if (shader != null)
        {
            _line.material = new Material(shader);
            _line.material.color = beamColor;
        }

        _line.useWorldSpace = true;
    }

    private void Update()
    {
        // Only show beam when a physical controller is active (not hand tracking)
        bool controllerActive =
            OVRInput.IsControllerConnected(OVRInput.Controller.RTouch) &&
            OVRInput.GetActiveController() != OVRInput.Controller.Hands;

        if (!controllerActive)
        {
            if (_line != null) _line.enabled = false;
            return;
        }

        if (_line != null) _line.enabled = true;

        Vector3 origin = transform.position;
        Vector3 direction = transform.forward;

        bool hitSomething = Physics.Raycast(origin, direction, out RaycastHit hit, maxRayLength);

        SubjectBox hitBox = null;
        if (hitSomething)
        {
            var tag = hit.collider.GetComponent<SubjectBoxTag>();
            if (tag != null && tag.box != null)
                hitBox = tag.box;
        }

        // Hover state
        if (hitBox != _hoveredBox)
        {
            if (_hoveredBox != null)
                SetBoxLineWidth(_hoveredBox, normalLineWidth);
            _hoveredBox = hitBox;
            if (_hoveredBox != null)
                SetBoxLineWidth(_hoveredBox, hoverLineWidth);
        }

        // Beam visual
        Vector3 endPoint = hitSomething ? hit.point : origin + direction * maxRayLength;
        _line.SetPosition(0, origin);
        _line.SetPosition(1, endPoint);
        if (_line.material != null)
            _line.material.color = hitBox != null ? beamHitColor : beamColor;

        // Trigger press
        if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger))
        {
            if (hitBox != null)
            {
                _selectedBox = hitBox;
                if (braceWs != null && int.TryParse(hitBox.subjectId, out int sid))
                    braceWs.SelectSubject(sid);
                OnSubjectSelected?.Invoke(hitBox);
            }
            else if (_selectedBox != null)
            {
                _selectedBox = null;
                if (braceWs != null)
                    braceWs.SelectSubject(null);
                OnSubjectSelected?.Invoke(null);
            }
        }
    }

    private void SetBoxLineWidth(SubjectBox box, float width)
    {
        if (box?.line != null)
        {
            box.line.startWidth = width;
            box.line.endWidth = width;
        }
    }
}
