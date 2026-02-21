using UnityEngine;

public class SelectableCube : MonoBehaviour
{
    public int cubeId;
    public string cubeInfo;

    private MeshRenderer _renderer;
    private Material _defaultMaterial;
    private Material _highlightMaterial;

    public void Initialize(int id, Material defaultMat, Material highlightMat)
    {
        cubeId = id;
        _renderer = GetComponent<MeshRenderer>();
        _defaultMaterial = defaultMat;
        _highlightMaterial = highlightMat;
        cubeInfo = $"Cube #{id}\nPosition: {transform.position:F1}\n" +
                   $"Velocity: {Random.Range(0.1f, 3.0f):F2} m/s\n" +
                   $"Joint Load: {Random.Range(20f, 180f):F0} N\n" +
                   $"FPPA: {Random.Range(-15f, 15f):F1} deg\n" +
                   $"Trunk Lean: {Random.Range(-10f, 10f):F1} deg";
    }

    public void Highlight()
    {
        if (_highlightMaterial != null)
            _renderer.material = _highlightMaterial;
    }

    public void Unhighlight()
    {
        if (_defaultMaterial != null)
            _renderer.material = _defaultMaterial;
    }

    public void Select() { }
    public void Deselect() { }
}
