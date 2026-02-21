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
        _line = GetComponent<LineRenderer>();
        if (_line == null)
            _line = gameObject.AddComponent<LineRenderer>();

        _line.positionCount = 2;
        _line.startWidth = beamWidth;
        _line.endWidth = beamWidth * 0.5f;
        _line.material = new Material(Shader.Find("Sprites/Default"));
        _line.material.color = beamColor;
        _line.useWorldSpace = true;
    }

    private void Update()
    {
        Vector3 origin = transform.position;
        Vector3 direction = transform.forward;

        bool hitSomething = Physics.Raycast(origin, direction, out RaycastHit hit, maxRayLength);

        // Look for a SubjectBoxTag on the hit collider
        SubjectBox hitBox = null;
        if (hitSomething)
        {
            var tag = hit.collider.GetComponent<SubjectBoxTag>();
            if (tag != null)
                hitBox = tag.box;
        }

        // Update hover state
        if (hitBox != _hoveredBox)
        {
            // Unhover previous
            if (_hoveredBox != null)
                SetBoxLineWidth(_hoveredBox, normalLineWidth);

            _hoveredBox = hitBox;

            // Hover new
            if (_hoveredBox != null)
                SetBoxLineWidth(_hoveredBox, hoverLineWidth);
        }

        // Update beam visual
        Vector3 endPoint = hitSomething ? hit.point : origin + direction * maxRayLength;
        _line.SetPosition(0, origin);
        _line.SetPosition(1, endPoint);
        _line.material.color = hitBox != null ? beamHitColor : beamColor;

        // Handle trigger press
        if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger))
        {
            if (hitBox != null)
            {
                _selectedBox = hitBox;

                // Tell the server to send detailed data for this subject
                if (braceWs != null && int.TryParse(hitBox.subjectId, out int sid))
                    braceWs.SelectSubject(sid);

                OnSubjectSelected?.Invoke(hitBox);
            }
            else
            {
                // Pressed on empty space — deselect
                if (_selectedBox != null)
                {
                    _selectedBox = null;

                    if (braceWs != null)
                        braceWs.SelectSubject(null);

                    OnSubjectSelected?.Invoke(null);
                }
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
