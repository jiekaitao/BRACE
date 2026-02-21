using System;
using UnityEngine;

public class ControllerRaycast : MonoBehaviour
{
    public static event Action<SelectableCube> OnCubeSelected;

    [SerializeField] private float maxRayLength = 10f;
    [SerializeField] private float beamWidth = 0.005f;
    [SerializeField] private Color beamColor = new Color(0.5f, 1f, 1f, 0.8f);
    [SerializeField] private Color beamHitColor = new Color(1f, 1f, 1f, 1f);

    private LineRenderer _line;
    private SelectableCube _hoveredCube;
    private SelectableCube _selectedCube;

    private void Start()
    {
        _line = GetComponent<LineRenderer>();
        if (_line == null)
            _line = gameObject.AddComponent<LineRenderer>();

        _line.positionCount = 2;
        _line.startWidth = beamWidth;
        _line.endWidth = beamWidth * 0.5f;
        // Sprites/Default is always included in builds and supports stereo instancing
        _line.material = new Material(Shader.Find("Sprites/Default"));
        _line.material.color = beamColor;
        _line.useWorldSpace = true;
    }

    private void Update()
    {
        Vector3 origin = transform.position;
        Vector3 direction = transform.forward;

        bool hitSomething = Physics.Raycast(origin, direction, out RaycastHit hit, maxRayLength);
        SelectableCube cube = hitSomething ? hit.collider.GetComponent<SelectableCube>() : null;

        // Update hover state
        if (cube != _hoveredCube)
        {
            if (_hoveredCube != null)
                _hoveredCube.Unhighlight();
            _hoveredCube = cube;
            if (_hoveredCube != null)
                _hoveredCube.Highlight();
        }

        // Update beam visual
        Vector3 endPoint = hitSomething ? hit.point : origin + direction * maxRayLength;
        _line.SetPosition(0, origin);
        _line.SetPosition(1, endPoint);
        _line.material.color = cube != null ? beamHitColor : beamColor;

        // Handle trigger press
        if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger))
        {
            if (cube != null)
            {
                // Deselect previous
                if (_selectedCube != null && _selectedCube != cube)
                    _selectedCube.Deselect();

                _selectedCube = cube;
                cube.Select();
                OnCubeSelected?.Invoke(cube);
            }
            else
            {
                // Pressed on empty space — deselect
                if (_selectedCube != null)
                {
                    _selectedCube.Deselect();
                    _selectedCube = null;
                    OnCubeSelected?.Invoke(null);
                }
            }
        }
    }
}
