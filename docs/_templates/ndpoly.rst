{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   .. automethod:: __call__

   .. rubric:: Methods

   .. autosummary::

      ~ndpoly.from_attributes
      ~ndpoly.isconstant
      ~ndpoly.todict
      ~ndpoly.tonumpy

   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::

      ~ndpoly.coefficients
      ~ndpoly.dtype
      ~ndpoly.exponents
      ~ndpoly.indeterminants
      ~ndpoly.keys
      ~ndpoly.names
      ~ndpoly.values
      ~ndpoly.KEY_OFFSET

   {% endif %}
   {% endblock %}
