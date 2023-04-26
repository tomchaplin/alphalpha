//! This crate implements the alpha and weak-alpha filtration of a set of points in the plane.
//! Both are filtrations of the Delaunay triangulation.
//! With the optional `lophat` feature, the crate also provides a function for computing the boundary matrix of the filtration.
//!
//! * The [alpha filtration](alpha_filtration) is constructed similarly to the Čech filtration.
//!   Grow balls of radius r around each point and intersect each ball with the corresponding Voronoi cell.
//!   This nerve of this collection of open sets is the alpha filtration at radius r.
//! * The [weak alpha filtration](weak_alpha_filtration) is a sub-filtration of the Vietoris-Rips filtration.
//!   Namely, at each filtration value r, the weak alpha filtration is equal to the Vietoris-Rips filtration intersected with the Delaunay triangulation.
//!
//! Filtrations of the Delauny triangulation are implemented as a [`DelaunayTriangulation`] from the [`spade`] crate.
//! The filtration value is stored in the [`data()`](spade::internals::DynamicHandleImpl::data) struct associated to each
//! [vertex](spade::handles::VertexHandle),
//! [undirected edge](spade::handles::UndirectedEdgeHandle) and
//! [inner face](spade::handles::FaceHandle).
//!
//! **WARNING:** To avoid unecessary square roots, the filtration times are squared from their theoretical value.

use std::collections::HashSet;

use spade::{
    handles::{FixedFaceHandle, FixedUndirectedEdgeHandle, FixedVertexHandle, InnerTag},
    DelaunayTriangulation, HasPosition, HierarchyHintGenerator, Point2, Triangulation,
};

#[cfg(feature = "lophat")]
mod lophat;
#[cfg(feature = "lophat")]
pub use crate::lophat::sparsify;

/// Represents a point in ℝ^2 that enters the filtration at the specified time.
pub struct FilteredPoint2 {
    /// The position of the point
    pub point: Point2<f64>,

    // This is here in case we define filtrations with non-zero vertex entry times later
    #[allow(dead_code)]
    /// The time at which the point enters the filtration
    pub filtration: f64,
}

impl HasPosition for FilteredPoint2 {
    type Scalar = <Point2<f64> as HasPosition>::Scalar;

    fn position(&self) -> Point2<Self::Scalar> {
        self.point.position()
    }
}

/// Represents an edge in ℝ^2 that enters the filtration at the specified time.
pub struct FilteredEdge {
    /// The time at which the edge enters the filtration
    pub filtration: f64,
}

impl Default for FilteredEdge {
    fn default() -> Self {
        Self {
            filtration: f64::NAN,
        }
    }
}

/// Represents a face in ℝ^2 that enters the filtration at the specified time.
pub struct FilteredFace {
    /// The time at which the face enters the filtration
    pub filtration: f64,
}

impl Default for FilteredFace {
    fn default() -> Self {
        Self {
            filtration: f64::NAN,
        }
    }
}

/// A Delaunay triangulation of a set of points in the plane, in which each point, edge and triangle additionaly carries the time at which it enters the filtration.
pub type DelaunayFiltration = DelaunayTriangulation<
    FilteredPoint2,
    (),
    FilteredEdge,
    FilteredFace,
    HierarchyHintGenerator<f64>,
>;

trait AlphaPropogateFaceExt {
    // Draw the a circle centred on the midpoint of `edge` with both edge-endpoints on the circumference
    // The edge is Gabriel wrt `vertex` iff vertex lies on or outside  the circle
    fn is_gabriel(&self, edge: FixedUndirectedEdgeHandle, vertex: FixedVertexHandle) -> bool;
    fn alpha_propogate_face(&mut self, face: FixedFaceHandle<InnerTag>);
}

// Edge should be one of the adjacent edges to face, so that the difference is exactly one vertex
fn get_none_edge_vertex(
    triangulation: &DelaunayFiltration,
    face: FixedFaceHandle<InnerTag>,
    edge: FixedUndirectedEdgeHandle,
) -> FixedVertexHandle {
    let face_vertices: HashSet<_> = triangulation.face(face).vertices().into_iter().collect();
    let edge_vertices: HashSet<_> = triangulation
        .undirected_edge(edge)
        .vertices()
        .into_iter()
        .collect();
    let mut difference = face_vertices.difference(&edge_vertices);
    difference
        .next()
        .expect("There should be some face vertex that is not on the edge")
        .fix()
}

impl AlphaPropogateFaceExt for DelaunayFiltration {
    // TODO: Change so that we don't check vertices that are on the edge
    fn is_gabriel(&self, edge: FixedUndirectedEdgeHandle, vertex: FixedVertexHandle) -> bool {
        let edge = self.undirected_edge(edge);
        let center = edge.center();
        let radius_2 = edge.length_2() / 4.0;
        let vertex_pos = self.vertex(vertex).position();
        let distance_2 = center.distance_2(vertex_pos);
        // True if this point lies on or outside the circle
        distance_2 >= radius_2
    }

    fn alpha_propogate_face(&mut self, face: FixedFaceHandle<InnerTag>) {
        let face_fil = self.face(face).data().filtration;
        // We have to make a list of fixed reference so we can mutate self
        // in order to propogate the filtration
        let edges = self
            .face(face)
            .adjacent_edges()
            .map(|e| e.as_undirected().fix());
        for edge in edges {
            let edge_fil = self.undirected_edge(edge).data().filtration;
            if !edge_fil.is_nan() {
                self.undirected_edge_data_mut(edge).filtration = face_fil.min(edge_fil);
            } else {
                let vertex = get_none_edge_vertex(self, face, edge);
                if !self.is_gabriel(edge, vertex) {
                    self.undirected_edge_data_mut(edge).filtration = face_fil;
                }
            }
        }
    }
}

/// Computes the non-truncated alpha filtration of a set of a points in the plane.
/// Filtration times are assigned according to the algorithm described in the [Gudhi documentation](https://gudhi.inria.fr/doc/latest/group__alpha__complex.html).
pub fn alpha_filtration(points: Vec<Point2<f64>>) -> DelaunayFiltration {
    // Build up the delaunay triangulation
    let filtered_points = points
        .into_iter()
        .map(|point| FilteredPoint2 {
            filtration: 0.0,
            point,
        })
        .collect();
    let mut triangulation = DelaunayFiltration::bulk_load(filtered_points)
        .expect("Should be able to build triangulation on point-set");
    // Distance to the circumcentre controls the entry time of each face.
    for face in triangulation.fixed_inner_faces() {
        let (_, square_circum_rad) = triangulation.face(face).circumcircle();
        triangulation.face_data_mut(face).filtration = square_circum_rad;
        // Propogate filtration values to relevant boundary edges
        triangulation.alpha_propogate_face(face);
    }
    // Add filtration values to edges that don't have one yet
    for edge in triangulation.fixed_undirected_edges() {
        let edge_fil = triangulation.undirected_edge(edge).data().filtration;
        if edge_fil.is_nan() {
            let length_2 = triangulation.undirected_edge(edge).length_2();
            let edge_fil = length_2 / 4.0;
            triangulation.undirected_edge_data_mut(edge).filtration = edge_fil;
        }
    }
    triangulation
}

/// Computes the non-truncated weak-alpha filtration of a set of a points in the plane.
/// As described by [Gitto](https://giotto-ai.github.io/gtda-docs/0.4.0/modules/generated/homology/gtda.homology.WeakAlphaPersistence.html), all simplices are given the same filtration value as they would be given in the Vietoris-Rips filtration.
pub fn weak_alpha_filtration(points: Vec<Point2<f64>>) -> DelaunayFiltration {
    // Build up the delaunay triangulation
    let filtered_points = points
        .into_iter()
        .map(|point| FilteredPoint2 {
            filtration: 0.0,
            point,
        })
        .collect();
    let mut triangulation = DelaunayFiltration::bulk_load(filtered_points)
        .expect("Should be able to build triangulation on point-set");
    // Each edge (i, j) enters the filtration at d(i, j)^2
    for edge in triangulation.fixed_undirected_edges() {
        let positions = triangulation.undirected_edge(edge).positions();
        let square_dist = positions[0].distance_2(positions[1]);
        triangulation.undirected_edge_data_mut(edge).filtration = square_dist;
    }
    // Each face enters the filtration as soon as all of its boundary edges appear
    for face in triangulation.fixed_inner_faces() {
        let boundary = triangulation.face(face).adjacent_edges();
        let max_entry = boundary
            .map(|edge| {
                let edge = triangulation.undirected_edge(edge.fix().as_undirected());
                let entry_time = edge.data().filtration;
                entry_time
            })
            .into_iter()
            .max_by(|a, b| {
                a.partial_cmp(b)
                    .expect("No edge should have a NaN filtration time")
            })
            .expect("There should be a maximum of the boundary entry times");
        triangulation.face_data_mut(face).filtration = max_entry;
    }
    triangulation
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn simple_square() -> Vec<Point2<f64>> {
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(0.0, 2.0),
            Point2::new(2.0, 2.0),
        ]
    }

    #[test]
    fn weak_alpha_simple_example() {
        let points = simple_square();
        let filtration = weak_alpha_filtration(points);
        // Triangulation with 5 edges and 2 triangles
        let n_edges = filtration.num_undirected_edges();
        assert_eq!(n_edges, 5);
        let n_faces = filtration.num_inner_faces();
        assert_eq!(n_faces, 2);
        // Each triangle enters at time 8.0 because
        // diagonal distance^2 from (2.0, 0.0) to (0.0, 2.0) is 8.0
        for face in filtration.fixed_inner_faces() {
            let face_time = filtration.face(face).data().filtration;
            assert_eq!(face_time, 8.0)
        }
    }

    // TODO: Find example that tests Gabriel propogation algorithm

    #[test]
    fn alpha_simple_example() {
        let points = simple_square();
        let filtration = alpha_filtration(points);
        // Triangulation with 5 edges and 2 triangles
        let n_edges = filtration.num_undirected_edges();
        assert_eq!(n_edges, 5);
        let n_faces = filtration.num_inner_faces();
        assert_eq!(n_faces, 2);
        // Each triangle enters at time 2.0 because
        // circum-centre of triangle in (0,0), (2,0), (0,2) is (1,1)
        // hence circum radius is \sqrt(2)
        for face in filtration.fixed_inner_faces() {
            let face_time = filtration.face(face).data().filtration;
            assert_eq!(face_time, 2.0)
        }
    }
}
